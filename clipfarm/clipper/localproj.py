"""AI videos made IN PLACE from a folder on this computer (no upload, nothing copied, nothing joined into one giant file).

Built for projects like 114 GB of footage: the originals are only ever READ.

    <your folder>/
        any videos (subfolders fine; a folder/file named "B-roll" = cutaway footage), notes.md, Music/ (songs to use)
        Output/          the finished video, v1, v2 ...        <- written here
        Final Cut/       the Final Cut Pro project              <- written here
        Notes/           plan.md, transcript.txt                <- written here
        .aivideo/        working files: transcripts, B-roll descriptions, the edit (plan.json), the rough cut

How a video gets made
    1. scan       every video gets a place on one virtual timeline (in shooting order); nothing is read except headers
    2. transcribe the speech: compressed audio (a few MB per hour) goes to OpenAI whisper-1, or Parakeet runs on this Mac
    3. look       B-roll files are looked at (a few frames each) and described by a vision model
    4. plan       the AI picks what to keep: story order for a vlog, or the best moments for a short cut of a lot of footage
    5. cutaways   B-roll is placed over the talking where it matches the words
    6. music      your song from Music/, or a licensed one that fits; "cut to the beat" nudges every cut onto the beat
    7. assemble   ONLY the kept stretches are cut out of the original files into a small rough cut
    8. finish     captions, music, animated title; a Final Cut project; copied into Output/ and Final Cut/

"Revise" takes plain words ("slightly too long, cut it shorter, bend to the beat, switch the song, add a better hook that
says XYZ") and turns them into those same steps.

With a footage catalog (catalog.roots, see catalog.py) step 5 also searches the whole library: the shots it finds join the project as
extra B-roll files placed far past the footage on the timeline (LIB_BASE), read where they are like everything else. A project made with
a "save to" folder writes its results and working files into <that folder>/<title>/ and only ever reads the footage folder.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

from ..core import ffmpeg
from ..core.log import get_logger
from . import audiosync as AS
from . import beats as BT
from . import logcolor as LC
from . import flow as FL
from . import live
from . import notes as NT
from . import broll as BR
from . import highlights as HL
from . import multicam as MC
from . import speakers as SP
from . import moments as M
from . import music as MU
from . import montage as MO
from . import story
from . import photos as PH
from . import tighten as TI
from . import transitions as TR
from . import transcript_edit as TE
from . import storage as ST
from . import avoid as AV
from .engine import PACES, Cancelled, Engine, remap_words, repace
from .library import safe
from .localai import LocalAI, estimate_cost

log = get_logger("clipper.localproj")
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
NOTES_MIN_S = 30.0                                       # a video shorter than this gets no chapters (notes.chapters)
# What a project folder holds that is NOT footage: Final Cut's own media and library, old exports, caches, our own results, songs and sound effects
SKIP_DIRS = {"output", "final cut", "notes", ".aivideo", "rejected", "music", "exports", "export", "renders", "render files", "transcoded media",
             "proxy media", "optimized media", "cache", "backup", "backups", "trash", "sfx", "sound effects", "sound fx", "soundfx", "sounds", "test shots", "tests", "test", "outtakes"}
SKIP_PATTERN = re.compile(r"(\.fcpbundle|\.fcpcache|\.imovielibrary|\.photoslibrary|\.app|^final cut|^render|^\.|proxy|optimized)", re.I)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".svg"}


def skipped(rel_parts) -> bool:
    """True if any folder on the way is one we don't read footage from."""
    return any(p.lower() in SKIP_DIRS or SKIP_PATTERN.search(p) for p in rel_parts)


# An earlier edit of the project sitting next to its footage ("Draft 1.mov", "Workshop Video V2.mov"): the same shots again, so
# using it as footage brings in repeats. Told apart by its name, unless it sits in a folder of camera files.
EDIT_NAME = re.compile(r"(?<![a-z0-9])(drafts?|v\d{1,2}|ver(?:sion)?[ -]*\d+|final|export(?:ed)?|rough[ -]*cut|first[ -]*cut|fine[ -]*cut|master|rendered|"
                       r"bounce|for (?:youtube|instagram|web))(?![a-z])", re.I)
CAMERA_DIR = re.compile(r"raw|footage|clip|a[ _-]?roll|b[ _-]?roll|camera|^cam\b|^cam[ _-]?[a-z0-9]$|^[a-z0-9][ _-]?cam$|card|dcim|private|interview", re.I)


def earlier_edit(rel: Path, folder: Path) -> bool:
    """True for a video that is an earlier edit of this project, not footage: named like one (draft, V2, final, export, rough cut,
    master...) or after the project itself, and not inside a folder of camera files."""
    if any(CAMERA_DIR.search(part) for part in rel.parts[:-1]):
        return False
    stem = re.sub(r"[_.]+", " ", rel.stem)
    if EDIT_NAME.search(stem):
        return True
    words = [w for w in re.split(r"[\s_-]+", folder.name.lower()) if len(w) > 2]
    return len(words) >= 2 and all(w in stem.lower() for w in words)


def _videos(folder: Path) -> tuple[list[Path], list[Path]]:
    """The footage videos in the folder, and the earlier edits left out (none left out if that would leave nothing)."""
    vids = [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith(".") and p.is_file()
            and not skipped(p.relative_to(folder).parts[:-1])]
    edits = [p for p in vids if earlier_edit(p.relative_to(folder), folder)]
    if len(edits) == len(vids):
        return vids, []
    return [p for p in vids if p not in edits], edits


def earlier_edits(folder: Path) -> list[str]:
    """The names of the earlier edits a scan of this folder leaves out (shown when the folder is chosen)."""
    return [str(p.relative_to(folder)) for p in _videos(folder)[1]]
CHUNK_S = 1200                                      # audio chunk for transcription (about 7 MB at 48 kbps)
MAX_EDGE = 1920                                     # the finished video is at most 1080p-class; the originals stay as they are
SENTENCE_LIMIT = 700                                # plain-words edits read the sentences of the CURRENT cut, so it must stay readable
LIB_BASE = 2_000_000.0                              # footage from the catalog sits here on the timeline: past the project's own (and its photos)
LIB_PHOTO_BASE = PH.VIRTUAL_BASE + 500_000.0        # ...and catalog photos here (the project's own photos count up from VIRTUAL_BASE)
LIB_SHOTS = 48                                      # candidate shots from the catalog offered to the cutaway planner
STOCK_BASE = 3_000_000.0                            # stock B-roll (Pixabay / Pexels, only when asked for) sits here: past the library footage
STOCK_PHOTO_BASE = PH.VIRTUAL_BASE + 700_000.0      # ...and stock photos here
STOCK_QUERIES = 10                                  # at most this many stock searches for one video
STOCK_SYSTEM = (
    "You plan B-roll for a video. For each passage of what is said, look at the B-roll the video already has (listed) and say whether one of "
    "those shots shows what the passage is about ('covered'). Where none does, write one search for a stock footage site (Pixabay, Pexels): "
    "2 to 6 words, concrete and literal, something a camera could film (for example 'students studying in a library', 'city traffic at "
    "night', 'hands typing on a laptop'). Never a person's or an organisation's name (the stock sites never filmed them), never the speaker "
    "talking. No search where the passage is covered or where nothing visual fits.")
STOCK_SCHEMA = {"type": "object", "properties": {"passages": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "covered": {"type": "boolean"}, "looks": {"type": "array", "items": {"type": "string"}}},
    "required": ["n", "covered", "looks"]}}}, "required": ["passages"]}


# ----------------------------------------------------------------------------------------- scanning
def _born(p: Path) -> float:
    st = p.stat()
    return float(getattr(st, "st_birthtime", 0) or st.st_mtime)


def scan(folder: Path) -> list[dict]:
    """Every video in the folder as a place on one virtual timeline, in the order they were shot: by the time the camera recorded in
    the file when every file has one (a copy from the card changes the file's date on the drive, never what is inside it), else by the
    file's date. Two cameras that both write C0001.MP4 get names from their folders (stable however the files are ordered)."""
    from .combine import _created
    from .glasses import _display_dims
    from .polish import _fps
    found, edits = _videos(folder)
    for p in edits:
        log.info(f"leaving out {p.relative_to(folder)}: an earlier edit of this project, not footage")
    probed = []
    for p in found:
        try:
            info = ffmpeg.probe(p)
            vs = ffmpeg.video_stream(info)
            dur = float(info.get("format", {}).get("duration") or 0)
        except Exception as e:  # noqa: BLE001 - a broken file is skipped, not fatal
            log.warning(f"skipping {p}: {e}")
            continue
        if not vs or dur < 0.5:
            continue
        from .foldergfx import video_alpha
        if video_alpha(vs):                                                 # see-through (ProRes 4444, Animation...): a graphic, never footage
            log.info(f"{p.relative_to(folder)} has see-through parts: used as a graphic, not as footage")
            continue
        probed.append((p, info, vs, dur))
    shot = [_created(info) for _p, info, _vs, _d in probed]
    if probed and all(shot):
        order = sorted(range(len(probed)), key=lambda k: (shot[k].timestamp(), probed[k][0].name))
    else:
        order = sorted(range(len(probed)), key=lambda k: (_born(probed[k][0]), probed[k][0].name))
    counts: dict[str, int] = {}
    for p, *_ in probed:
        counts[p.name] = counts.get(p.name, 0) + 1
    files, t, seen = [], 0.0, set()
    for k in order:
        p, info, vs, dur = probed[k]
        rel = p.relative_to(folder)
        name = p.name if counts[p.name] == 1 else " - ".join(rel.parts)        # "Cam A - C0001.MP4": the same whichever is scanned first
        base, n = name, 2
        while name in seen:
            name = f"{Path(base).stem} ({n}){Path(base).suffix}"
            n += 1
        seen.add(name)
        w, h = _display_dims(vs)
        rate = _fps(vs)
        files.append({"tc": timecode_s(info, rate), "tc_df": timecode_df(info), "name": name, "path": str(p.resolve() if p.is_symlink() else p), "rel": str(rel),
                      "size": p.stat().st_size, "mtime": int(p.stat().st_mtime),
                      "dur": round(dur, 3), "start": round(t, 3), "end": round(t + dur, 3), "w": int(w), "h": int(h),
                      "fps": float(rate), "audio": ffmpeg.audio_stream(info) is not None,
                      "broll": BR.is_named(name) or any(BR.is_named(part) for part in rel.parts[:-1])})
        t += dur
    return files


def seg(words: list[dict]) -> list[M.Sentence]:
    """The transcript as whole thoughts, the units the editor picks.

    With punctuation (Parakeet, and OpenAI transcripts made since the punctuation is put back): a sentence ends where the speaker ended
    it; a very long one is split at a comma, and a long silence (a real stop) also ends one. Without punctuation (older OpenAI transcripts)
    the old rule stands unchanged - a thought ends at a pause of half a second, or after about 22 words - so an edit made from such a
    transcript keeps meaning the same sentences."""
    from .openai_audio import has_punctuation
    out: list[M.Sentence] = []
    cur: list[dict] = []
    punct = len(words) >= 30 and has_punctuation(words)
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 99
        if punct:
            end = (M.END.search(w["text"]) or gap >= 1.2 or len(cur) >= 60 or nxt is None
                   or (len(cur) >= 32 and re.search(r"[,;:]['\"”’)]*$", w["text"])))
        else:
            starts = bool(nxt) and nxt["text"][:1].isupper() and nxt["text"].strip(".,!?'\"") in STARTERS and gap >= 0.2     # a capital + a sentence-opening word
            end = M.END.search(w["text"]) or gap >= 0.5 or len(cur) >= 22 or nxt is None or (starts and len(cur) >= 4)
        if end:
            out.append(M.Sentence(len(out), cur[0]["start"], cur[-1]["end"], " ".join(x["text"] for x in cur).strip()))
            cur = []
    return out


STARTERS = {"And", "But", "So", "Now", "Then", "The", "We", "They", "It", "He", "She", "How", "What", "When", "Why", "Yes", "No", "Okay", "Well",
            "That", "This", "My", "Our", "You", "There", "If", "Because", "Also", "Just", "Like", "Honestly", "Actually", "Um", "Uh", "Oh"}


def _timecode(info: dict) -> str:
    """The clip's start timecode, but ONLY if Final Cut can read it: a QuickTime timecode track (codec tag "tmcd"). Sony cameras (the
    C0001.MP4 files, and the "mpro" files here) keep their timecode in a separate "rtmd" metadata stream, which ffprobe shows as a
    timecode tag but Final Cut does not read - it sees those files as starting at 00:00:00:00. Telling it they start at 03:51:50 puts every
    edit past the end of the media ("an edit with no respective media"). So for those files the answer is "" and the clip starts at 0."""
    for st in info.get("streams") or []:
        if str(st.get("codec_tag_string", "")).lower() == "tmcd":
            tc = (st.get("tags") or {}).get("timecode") or (info.get("format") or {}).get("tags", {}).get("timecode")
            if tc:
                return str(tc)
    return ""


def timecode_df(info: dict) -> bool:
    """Drop-frame timecode (29.97 / 59.94 cameras write it with a ';')."""
    return bool(re.match(r"^\d+:\d+:\d+[;.]\d+$", _timecode(info)))


def timecode_s(info: dict, fps) -> float:
    """The clip's embedded start timecode as real media time in seconds (cameras that record time-of-day timecode), or 0. Final Cut
    places a clip by its timecode, so the project must say exactly where each one starts, or the clips land in the wrong places.

    Timecode COUNTS FRAMES at the nominal rate (24 for 23.976, 30 for 29.97): 03:51:50:08 at 23.976 is 333,848 frames, which is
    333,848 x 1001/24000 = 13,924.24 s - not 13,910.33 s as the label reads. Drop-frame timecode skips labels to stay near the clock."""
    m = re.match(r"^(\d+):(\d+):(\d+)([:;.])(\d+)$", _timecode(info))
    if not m:
        return 0.0
    h, mi, s, sep, fr = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
    rate = Fraction(fps).limit_denominator(1001) if fps else Fraction(30)
    nominal = max(1, round(float(rate)))
    frames = (h * 3600 + mi * 60 + s) * nominal + fr
    if sep in ";." and nominal in (30, 60):                          # drop-frame: 2 (or 4) labels skipped every minute but each tenth
        minutes = h * 60 + mi
        frames -= (nominal // 15) * (minutes - minutes // 10)
    return round(float(frames / rate), 6)


def scan_graphics(folder: Path) -> list[dict]:
    """Pictures in the project (a logo, brand art, product photos): in the folder itself and in graphics-like subfolders. Never in the
    footage / Final Cut / export folders, and never our own results."""
    out = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT or p.name.startswith(".") or not p.is_file():
            continue
        rel = p.relative_to(folder)
        if skipped(rel.parts[:-1]):
            continue
        if len(rel.parts) > 1 and not re.search(r"graphic|logo|brand|asset|image|photo|art|title|overlay", rel.parts[-2], re.I):
            continue                                       # a picture buried in some other folder is probably not meant for the video
        try:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001 - svg or unreadable: keep it, size unknown
            w = h = 0
        out.append({"name": p.name, "path": str(p), "size": p.stat().st_size, "w": w, "h": h})
    return out[:12]


def scan_audio(folder: Path, extra: list[str] | None = None) -> list[dict]:
    """Separate audio recordings (a recorder, a lav mic): in the folder (outside Music/ and the result folders), in the folder just above
    it, and anywhere you point at (files or folders). A recording that has nothing to do with the footage simply won't match by sound."""
    found: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AS.AUDIO_EXT or p.name.startswith(".") or not p.is_file():
            continue
        rel = p.relative_to(folder)
        if skipped(rel.parts[:-1]):
            continue
        found.append(p)
    try:
        found += [p for p in sorted(folder.parent.iterdir()) if p.is_file() and p.suffix.lower() in AS.AUDIO_EXT and not p.name.startswith(".")]
    except OSError:
        pass
    for e in extra or []:
        q = Path(str(e).strip()).expanduser()
        if q.is_file() and q.suffix.lower() in AS.AUDIO_EXT:
            found.append(q)
        elif q.is_dir():
            found += [p for p in sorted(q.rglob("*")) if p.is_file() and p.suffix.lower() in AS.AUDIO_EXT and not p.name.startswith(".")]
    out, seen = [], set()
    for p in found:
        if str(p) in seen:
            continue
        seen.add(str(p))
        try:
            info = ffmpeg.probe(p)
            dur = float(info.get("format", {}).get("duration") or 0)
        except Exception:  # noqa: BLE001
            continue
        if dur >= 5:
            au = ffmpeg.audio_stream(info) or {}
            sr = int(au.get("sample_rate") or 48000)
            try:                                                   # a recorder's start stamp (BWF time reference, samples since midnight):
                tc = int(((info.get("format") or {}).get("tags") or {}).get("time_reference") or 0) / sr     # Final Cut places it by that
            except (TypeError, ValueError):
                tc = 0.0
            out.append({"name": p.name, "path": str(p), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime), "dur": round(dur, 1),
                        "tc": round(tc, 6), "ch": int(au.get("channels") or 2)})
    return out


def summarize(folder: Path, cfg=None, files: list[dict] | None = None, target_min: float = 0.0, transcriber: str = "parakeet") -> dict:
    """What is in the folder and what making a video from it should cost - shown BEFORE anything is sent anywhere."""
    files = files if files is not None else scan(folder)
    talk_s = sum(f["dur"] for f in files if not f["broll"] and f["audio"])
    broll = [f for f in files if f["broll"]]
    model = str(cfg.get("clipper.ai_videos.openai.model", "gpt-4.1-mini")) if cfg else "gpt-4.1-mini"
    est = estimate_cost(model, talk_s, len(broll), 2, transcriber,
                        str(cfg.get("clipper.ai_videos.openai.cheap_model", "gpt-4.1-nano")) if cfg else "gpt-4.1-nano")
    return {"folder": str(folder), "files": len(files), "gb": round(sum(f["size"] for f in files) / 1e9, 2),
            "hours": round(sum(f["dur"] for f in files) / 3600, 2), "talk_hours": round(talk_s / 3600, 2), "broll_files": len(broll),
            "notes": bool(read_notes(folder)), "music": len(MU.own_tracks(folder / "Music")), "estimate": est,
            "audio_files": len(scan_audio(folder)), "left_out": earlier_edits(folder)}


VERSIONED = re.compile(r"^(?!\.).+ v(\d+)\.(mp4|srt|fcpxml)$")


def older_versions_to_trash(folder: Path, version: int) -> list[Path]:
    """A new version is saved: the older videos and captions (Output/) and .fcpxml files (the folder itself) go to the Trash, so the
    folder holds only the newest (2026-11-06: "there's three versions of this video and it didn't delete the old ones"). The Trash, not
    gone: they can be put back. The Final Cut/<name> vN folders stay: a project already imported in Final Cut points at them."""
    old = [f for d in (folder / "Output", folder) if d.is_dir() for f in d.iterdir()
           if f.is_file() and (m := VERSIONED.match(f.name)) and int(m.group(1)) < version and (d == folder) == (m.group(2) == "fcpxml")]
    if old:
        to_trash(old)
    return old


def to_trash(paths: list[Path]) -> None:
    """Into the Trash (macOS 15's `trash`, else the Finder); never a hard delete. A failure leaves the files where they are."""
    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        return
    try:
        if shutil.which("trash") or Path("/usr/bin/trash").exists():
            subprocess.run([shutil.which("trash") or "/usr/bin/trash", *paths], check=True, capture_output=True, timeout=60)
        else:
            items = ", ".join(f'POSIX file "{p}"' for p in paths)
            subprocess.run(["osascript", "-e", f'tell application "Finder" to delete {{{items}}}'], check=True, capture_output=True, timeout=60)
        log.info(f"older version(s) moved to the Trash: {', '.join(Path(p).name for p in paths)}")
    except Exception as e:  # noqa: BLE001 - the folder keeps them; nothing is lost
        log.warning(f"couldn't move older versions to the Trash: {e}")


def home_of(proj: dict) -> Path:
    """Where a project writes: its "save to" folder (fixed when it was made), else the footage folder itself."""
    return Path(proj.get("home") or proj["folder"])


def read_notes(folder: Path) -> str:
    for n in ("notes.md", "notes.txt", "Notes.md", "Notes.txt", "NOTES.md"):
        p = folder / n
        if p.is_file():
            return p.read_text(errors="replace").strip()[:3000]
    return ""


# ----------------------------------------------------------------------------------------- the work folder
class State:
    """The project's working files in <folder>/.aivideo (JSON, written atomically)."""

    def __init__(self, folder: Path):
        self.dir = folder / ".aivideo"
        self.dir.mkdir(parents=True, exist_ok=True)              # (a "save to" folder that was cleared away is made again)

    def get(self, name: str, default=None):
        p = self.dir / name
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return default

    def put(self, name: str, data) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.dir / name)

    def snapshot(self, name: str, data, reason: str) -> Path:
        """Keep a recoverable copy of an edit before replacing it."""
        history = self.dir / "backups"
        history.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^a-z0-9_-]+", "-", reason.lower()).strip("-") or "revision"
        path = history / f"{Path(name).stem}-{stamp}-{safe}.json"
        n = 2
        while path.exists():
            path = history / f"{Path(name).stem}-{stamp}-{safe}-{n}.json"
            n += 1
        path.write_text(json.dumps(data, indent=1))
        return path


class LocalEngine(Engine):
    """The normal Engine, with every AI question going to OpenAI (see localai.py)."""

    def __init__(self, cfg, store, ai: LocalAI):
        super().__init__(cfg, store)
        self.ai = ai

    def router_for_now(self):
        return self.ai, "openai"

    def creative_router(self):
        return self.ai

    def vision_router(self):
        return self.ai, "openai"


# ----------------------------------------------------------------------------------------- the runner
class LocalRunner:
    """Runs local projects one at a time on a background thread; progress goes through the project's job so the page shows it."""

    def __init__(self, cfg, store, projects):
        self.cfg, self.store, self.projects = cfg, store, projects
        self.lock = threading.Lock()
        self.review_lock = threading.Lock()
        self.running: str | None = None
        self.catalog = None                                          # the footage catalog (set by the server; else made on first use)

    def enabled(self) -> bool:
        return bool(self.cfg.get("clipper.ai_videos.local_projects", False))

    def start(self, pid: str, kind: str, **kw) -> None:
        if not self.lock.acquire(blocking=False):
            if self.running == pid:
                raise ValueError("this video is being made right now")
            job = self.projects.get(pid)["job"]                      # another one is being made: this one waits its turn (resume() starts it,
            self.store.update(job, status="queued", stage="Waiting for the video being made now", error=None, cancel=False,   # oldest first, when
                              run={"kind": kind, "kw": kw, "at": time.time()})                                                 # the one in hand is done)
            return
        self.running = pid
        self._tok = tok = getattr(self, "_tok", 0) + 1               # (which run this is: a run that was stopped for good must not touch anything afterwards)
        self.cancel_notes(pid)
        job = self.projects.get(pid)["job"]
        self.store.update(job, run={"kind": kind, "kw": kw, "at": time.time(), "units": self._units_of(pid)})   # what it is doing (a quit app carries on with it) and since when

        def run():
            stop = threading.Event()
            threading.Thread(target=self._tick, args=(job, stop), daemon=True, name=f"local-tick-{pid}").start()
            ok = False
            try:
                self.store.update(job, status="transcribing", stage="Starting", error=None, cancel=False, detail="", progress=0.0)
                getattr(self, kind)(pid, **kw)
                self.store.update(job, run=None)
                ok = True
            except Cancelled:
                if tok == self._tok:
                    self.store.update(job, status="cancelled", stage="Stopped", run=None)
            except Exception as e:  # noqa: BLE001 - shown on the page; finished steps are cached for the next try
                if tok == self._tok:
                    log.exception(f"local project {pid} failed")
                    self.store.update(job, status="failed", stage="Failed", error=f"{type(e).__name__}: {e}"[:800], run=None)
                else:
                    log.info(f"local project {pid}: the run that was stopped ended with {type(e).__name__}")
            finally:
                stop.set()
                if tok != self._tok:                                  # stopped for good: it was already given up on (hard_stop), nothing is left to do
                    return
                try:
                    self.store.update(job, detail="")
                except Exception:  # noqa: BLE001
                    pass
                self.cur_ai = None
                self.running = None
                self.lock.release()
                ST.after_run(self, pid, kind, ok)               # (a video rendered again is no longer marked as removed to save space)
                TE.after_run(self, pid, kind, ok)               # (a transcript edit queued while this was being made starts now)
        threading.Thread(target=run, daemon=True, name=f"local-{pid}").start()

    # -- Stop: asks the run to stop at its next step, and if it is stuck (an AI call, an encode that never ends) ends it for good --------------
    @staticmethod
    def _render_children() -> list[int]:
        """Processes this engine started for a render (ffmpeg, the AI command lines), not the catalogue's background ones (nice 15) or the speech worker."""
        import os as _os
        try:
            out = subprocess.run(["ps", "-axo", "pid=,ppid=,ni=,comm="], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        rows = []
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
                rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
        kids, frontier = set(), {_os.getpid()}
        while frontier:
            nxt = {pid for pid, ppid, _, _ in rows if ppid in frontier and pid not in kids}
            kids |= nxt
            frontier = nxt
        names = ("ffmpeg", "ffprobe", "claude", "codex", "sips")
        return [pid for pid, _, ni, comm in rows if pid in kids and ni != "15" and any(comm.rsplit("/", 1)[-1].startswith(n) for n in names)]

    def hard_stop(self, pid: str, grace: float = 3.0) -> dict:
        """The Stop button. The run is asked to stop at its next step; one that is still going after `grace` seconds (waiting on an AI answer,
        an encode, something frozen) has its ffmpeg / AI processes killed, and one that is STILL there is given up on: the project is marked
        stopped, and the run, when it ever ends, changes nothing."""
        import os as _os
        import signal as _signal
        job = self.projects.get(pid)["job"]
        self.store.update(job, cancel=True)
        self.cancel_notes(pid)
        if self.running != pid:
            if self.store.get(job).get("status") == "queued":
                self.store.update(job, status="cancelled", stage="Stopped", run=None, detail="")
            return {"stopping": False}

        def reaper():
            time.sleep(grace)
            if self.running != pid:
                return
            for sig in (_signal.SIGTERM, _signal.SIGKILL):
                for kid in self._render_children():
                    try:
                        _os.kill(kid, sig)
                    except OSError:
                        pass
                time.sleep(2.0)
                if self.running != pid:
                    return
            log.warning(f"local project {pid}: Stop gave up on a run that would not end")
            self._tok = getattr(self, "_tok", 0) + 1                  # (the old run changes nothing from here on)
            self.store.update(job, status="cancelled", stage="Stopped", run=None, detail="", error=None, progress=0.0)
            self.cur_ai = None
            self.running = None
            try:
                self.lock.release()
            except RuntimeError:
                pass
        threading.Thread(target=reaper, daemon=True, name=f"local-stop-{pid}").start()
        return {"stopping": True}

    # -- the AI-written notes (YouTube chapters + description): after the video is done, one at a time, never able to change its status ----
    def _notes_state(self):
        if not hasattr(self, "_notes"):
            self._notes, self._notes_gate = {}, threading.Lock()
        return self._notes

    def start_notes(self, pid: str, out_words: list, dur: float, notes: Path, credit_lines: list, brief: str) -> None:
        ents = self._notes_state()
        self.cancel_notes(pid)
        cancel = threading.Event()

        def work():
            with self._notes_gate:                                           # one at a time: a slow account call must not pile up
                if cancel.is_set():
                    return
                state, chapters_text = "failed", ""
                try:
                    gl = self.cfg.get("clipper.glossary") or self.cfg.get("glossary") or []
                    gl = [x.strip() for x in (gl.splitlines() if isinstance(gl, str) else gl) if str(x).strip()] if isinstance(gl, (str, list)) else []
                    chs, summary = NT.chapters(LocalAI(self.cfg), out_words, dur, gl, brief)
                    if cancel.is_set():                                      # a newer render of this project took over: this answer is stale
                        return
                    chapters_text = NT.chapters_txt(chs)
                    if chapters_text:
                        (notes / "Chapters.txt").write_text(chapters_text)
                    (notes / "Description.txt").write_text("\n\n".join(x for x in (summary, ("Chapters\n" + chapters_text.strip()) if chapters_text else "",
                                                                                 "\n\n".join(credit_lines)) if x) + "\n")
                    state = "done"
                except Exception as e:  # noqa: BLE001 - chapters are a bonus, never a reason to touch the video's status
                    log.warning(f"chapters failed: {e}")
                if cancel.is_set():
                    return
                try:
                    cur = self.projects.get(pid).get("notes_info") or {}
                    self.projects.update(pid, notes_info={**cur, "chapters": chapters_text, "notes_state": state})
                except Exception:  # noqa: BLE001
                    log.warning("could not save the chapters note")
                finally:
                    ents.pop(pid, None)
        t = threading.Thread(target=work, daemon=True, name=f"notes-{pid}")
        ents[pid] = (t, cancel)
        t.start()

    def cancel_notes(self, pid: str, wait: float = 0.0) -> None:
        """A new run of this project: the notes thread still writing for the old video is told its answer is stale (it drops it)."""
        ent = self._notes_state().pop(pid, None)
        if ent:
            ent[1].set()
            if wait:
                ent[0].join(wait)

    def notes_alive(self, pid: str) -> bool:
        ent = self._notes_state().get(pid)
        return bool(ent and ent[0].is_alive())

    def transcript_edit(self, pid: str) -> None:
        """Runner action: the edits staged in the transcript panel are applied to the plan and the video renders again (transcript_edit.py)."""
        TE.run_apply(self, pid)

    def transcript_undo(self, pid: str) -> None:
        TE.run_undo(self, pid)

    TICK_S = 3.0
    CREEP_S = 240.0                     # a long step's bar moves 63% of the way to its creep limit in this many seconds

    def _tick(self, job: str, stop: threading.Event) -> None:
        """While a video is made: the job's `detail` says which AI call it waits on (LocalAI.now(), or that it is offline), and during a long
        step the bar creeps on (asymptotically, a few percent at most) so it never sits still for 20 minutes (a workshop shoot, 59%)."""
        last = None
        while not stop.wait(self.TICK_S):
            try:
                ai = getattr(self, "cur_ai", None)
                detail = ""
                if ai is not None:
                    detail = ai.now() or ""
                    if getattr(ai, "offline_note", ""):
                        detail = (ai.offline_note + (" · " + detail if detail else ""))
                upd: dict = {}
                if detail != last:
                    upd["detail"], last = detail, detail
                base = getattr(self, "_pbase", None)
                if base and base[0] == job:
                    shown = creep(base[1], time.time() - base[2], self.CREEP_S)
                    cur = float(self.store.get(job).get("progress") or 0.0)
                    if shown > cur + 0.001:
                        upd["progress"] = round(shown, 4)
                if upd:
                    self.store.update(job, **upd)
            except Exception:  # noqa: BLE001 - a progress line never stops a render
                pass

    def _units_of(self, pid: str):
        """Footage minutes of a project (from its scan, when it has been scanned): what the stage times are scaled by (progress.py)."""
        try:
            files = State(home_of(self.projects.get(pid))).get("files.json") or []
            m = sum(float(f.get("dur") or 0) for f in files if not f.get("photo")) / 60.0
            return round(m, 2) or None
        except Exception:  # noqa: BLE001
            return None

    def _sweep_notes(self) -> None:
        """A project whose notes say 'writing' but no thread is writing (the app quit meanwhile) is shown as failed, not 'writing' for ever."""
        try:
            for p in self.projects.list():
                ni = p.get("notes_info") or {}
                if ni.get("notes_state") == "writing" and not self.notes_alive(p["id"]):
                    self.projects.update(p["id"], notes_info={**ni, "notes_state": "failed"})
        except Exception:  # noqa: BLE001
            pass

    def resume(self) -> str | None:
        """A long video that was being made when the app quit (requeue_interrupted put it back as queued, with what it was doing) starts
        again, one at a time. Its finished steps are cached; an edit already decided is rendered as it was, not planned again."""
        self._sweep_notes()
        if self.running:
            return None
        for p in reversed(self.projects.list()):                        # (oldest first)
            try:
                j = self.store.get(p["job"])
            except KeyError:
                continue
            r = j.get("run") or {}
            if j.get("status") != "queued" or not r.get("kind") or not hasattr(self, str(r["kind"])):
                continue
            kind, kw = r["kind"], dict(r.get("kw") or {})
            if kind == "make" and p.get("pending_render"):
                kind, kw = "rerender", {}
            try:
                self.start(p["id"], kind, **kw)
                log.info(f"resumed {p['id']} ({kind}) after a restart")
                return p["id"]
            except ValueError:                                              # (another one just started)
                return None
        return None

    def keep_resuming(self, every: float = 5.0) -> None:
        def loop():
            while True:
                time.sleep(every)                                        # (after the worker has put interrupted ones back in the queue)
                try:
                    self.resume()
                except Exception:  # noqa: BLE001
                    log.exception("resuming a long video failed")
        threading.Thread(target=loop, name="local-resume", daemon=True).start()

    # -- helpers -----------------------------------------------------------------------------
    def _setup(self, pid: str):
        proj = self.projects.get(pid)
        folder = Path(proj["folder"])
        if not folder.is_dir():
            raise RuntimeError(f"the folder {folder} isn't there (is its drive plugged in?)")
        ai = LocalAI(self.cfg)
        ok, why = ai.available()
        if not ok:
            raise RuntimeError(why)
        engine = LocalEngine(self.cfg, self.store, ai)
        self.live_job = proj["job"]                                      # what the page's live view follows (live.py)
        self.cur_ai = ai                                                 # (the progress ticker asks it what it is waiting on)
        return proj, folder, State(home_of(proj)), ai, engine

    def _live(self, kind: str, **data) -> None:
        if getattr(self, "live_job", None):
            live.emit(self.store, self.live_job, kind, **data)

    def _live_lines(self, plan: dict, sents) -> None:
        """The live view: the lines the edit keeps, in order."""
        by = {x.idx: x for x in sents}
        for it in [it for it in plan.get("items") or [] if it.get("kind") == "talk"][:40]:
            ss = [by[i] for i in range(it["a"], it["b"] + 1) if i in by]
            if ss:
                self._live("line", text=" ".join(x.text for x in ss)[:280], start=round(ss[0].start, 1), end=round(ss[-1].end, 1))

    def _live_broll(self, cutaways: list[dict], pool: list[dict], files: list[dict]) -> None:
        """The live view: each B-roll shot placed over the talking, with a frame of it (made in the background)."""
        job = getattr(self, "live_job", None)
        if not job:
            return
        by_name = {f["name"]: f for f in files}

        def run():
            for c in cutaways[:40]:
                sc = pool[c["scene"]] if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(pool) else {}
                f = by_name.get(sc.get("file") or "")
                pic = ""
                if sc.get("photo"):
                    try:
                        pic = live.frame(self.store, job, Path(sc["photo"]).read_bytes())
                    except OSError:
                        pic = ""
                elif f and f.get("path"):
                    pic = live.frame(self.store, job, path=f["path"], t=max(0.0, float(sc.get("start", f["start"])) - float(f["start"]) + 0.5))
                # (what "Don't use this" files it under: the file as the edit calls it, and the stretch in that file; a library shot also carries
                #  what the page's player needs to play it)
                off = float(f["start"]) if f and not sc.get("photo") else 0.0
                a0, b0 = round(float(sc.get("start", 0)) - off, 2), round(float(sc.get("end", 0)) - off, 2)
                lib = {}
                if f and f.get("library") and not sc.get("photo") and f.get("path"):
                    try:
                        from .catalog import rid
                        rec = (self.catalog.load(rid(f["path"])) if self.catalog is not None else None) or {}
                        lib = {"id": rid(f["path"]), "dur": f.get("dur"), "step": rec.get("step"), "frames": rec.get("frames"), "rel": f.get("rel")}
                    except Exception:  # noqa: BLE001 - the picture alone still opens
                        lib = {}
                live.emit(self.store, job, "broll", f=pic, at=round(float(c.get("at") or 0), 1), length=round(float(c.get("length") or 0), 1),
                          what=str(sc.get("what") or "")[:160], of=str(sc.get("file") or "").replace("Library - ", "Library: ").replace("Stock - ", "Stock: "),
                          name=str(sc.get("file") or ""), start=a0, end=b0, photo=bool(sc.get("photo")), lib=lib or None)
        live.later(run)

    def library(self):
        """The footage catalog, when this Mac has one (catalog.roots): shared with the page's search, so the picture model loads once."""
        if self.catalog is None and self.cfg.get("catalog.roots"):
            from .catalog import Catalog
            self.catalog = Catalog(self.cfg)
        return self.catalog if self.catalog is not None and self.catalog.enabled() else None

    def add_library(self, files: list[dict], words: list[dict], scenes: dict, ai: LocalAI, st: State, brief: str, progress) -> tuple[list[dict], dict]:
        """B-roll from the whole footage library for what is said (catalog.shots_for), as extra files on the timeline past LIB_BASE
        (photos as photo scenes). Remembered in library.json, so a re-render or a change uses the same shots."""
        files = [f for f in files if not f.get("library")]
        cat = self.library()
        if cat is None:
            return files, scenes
        shots = AV.drop_found(AV.load(st), cat.shots_for(ai, seg(words), brief, limit=LIB_SHOTS, progress=progress))   # (shots ruled out stay out)
        if getattr(self, "live_job", None):
            def show(found=list(shots[:24])):
                for s in found:
                    try:
                        pic = live.frame(self.store, self.live_job, cat.thumb(s["id"], int(s.get("k") or 0)).read_bytes())
                    except (OSError, KeyError, TypeError, ValueError):
                        pic = ""
                    # (what the page's player needs to open it: the catalog id, the stretch, the stills; and the name "Don't use this" files it under)
                    live.emit(self.store, self.live_job, "libshot", f=pic, what=str(s.get("what") or "")[:160], rel=s.get("rel"),
                              query=s.get("query"), media=s.get("kind"), id=s.get("id"), start=s.get("start"), end=s.get("end"), k=s.get("k"),   # (not kind=: that is the event's own)
                              dur=s.get("dur"), step=s.get("step"), frames=s.get("frames"), name=f"Library - {s.get('rel')}")
            live.later(show)
        lib, by_id, lib_scenes, photos = [], {}, {}, []
        t = LIB_BASE
        for s in shots:
            q = float(s.get("quality") or 6.0)
            if s["kind"] == "photo":
                a = LIB_PHOTO_BASE + 10.0 * len(photos)
                photos.append({"start": a, "end": a + PH.SCENE_S, "what": f"(still photo) {s['what']}", "quality": max(6.0, q),
                               "file": f"Library - {s['rel']}", "photo": s["path"], "library": True})
                continue
            f = by_id.get(s["id"])
            if f is None:
                dur = float(s["dur"])
                f = {"name": f"Library - {s['rel']}", "path": s["path"], "rel": s["rel"], "size": s.get("size"), "mtime": s.get("mtime"),
                     "dur": dur, "start": round(t, 3), "end": round(t + dur, 3), "w": int(s.get("w") or 1920), "h": int(s.get("h") or 1080),
                     "fps": float(s.get("fps") or 30.0), "audio": bool(s.get("audio")), "tc": float(s.get("tc") or 0.0),
                     "tc_df": bool(s.get("tc_df")), "broll": True, "library": True}
                by_id[s["id"]] = f
                lib.append(f)
                t += dur + 10.0
            lib_scenes.setdefault(f["name"], []).append({"start": round(f["start"] + s["start"], 2), "end": round(f["start"] + s["end"], 2),
                                                         "what": s["what"], "quality": q})
        st.put("library.json", {"files": lib, "scenes": lib_scenes, "photos": photos,
                                "found": [{k: s.get(k) for k in ("rel", "start", "end", "what", "query", "score")} for s in shots]})
        if shots:
            log.info(f"{len(shots)} shot(s) from the footage library offered as B-roll ({len(lib)} video files, {len(photos)} photos)")
        return files + lib, {**scenes, **lib_scenes}

    def stock_dir(self, proj: dict) -> Path:
        """Where downloaded stock B-roll is kept: the app's own Stock folder (<home>/Stock, shared by every project, so a shot is downloaded
        once), or on the personal machines the project's .aivideo/stock."""
        import os
        if self.cfg.get("clipper.app") == "studio" or os.environ.get("VIDEOBUILDER_HOME"):
            from .studio import home
            return home() / "Stock"
        return home_of(proj) / ".aivideo" / "stock"

    def add_stock(self, files: list[dict], words: list[dict], scenes: dict, ai: LocalAI, st: State, brief: str, proj: dict,
                  progress) -> tuple[list[dict], dict]:
        """Stock B-roll (only when asked for: the `stock` option) for the passages nothing of yours shows: the folder's B-roll and photos and
        the footage library come first. Pixabay and Pexels are searched (videos; a photo when no video fits) with the editor's own
        description of each uncovered passage, and the best result of each search is offered to the cutaway planner like any B-roll: a
        file on the timeline past STOCK_BASE (photos as photo scenes). Nothing is downloaded here - only the shots the edit places are
        (fetch_stock). Remembered in stock.json, so a re-render or a change offers the same shots. No key on this Mac: stock is left out."""
        from .. import stock as SK
        files = [f for f in files if not f.get("stock")]
        scenes = {k: v for k, v in scenes.items() if not k.startswith("Stock - ")}
        empty = {"files": [], "scenes": {}, "photos": [], "found": []}
        if not SK.keys():
            log.info("stock B-roll was asked for, but this Mac has no Pixabay or Pexels key: the video is made without it")
            st.put("stock.json", empty)
            return files, scenes
        passages, cur = [], []
        for s in seg(words):
            cur.append(s)
            if s.end - cur[0].start >= 18.0:
                passages.append(cur)
                cur = []
        if cur:
            passages.append(cur)
        have = [s for v in scenes.values() for s in v]
        try:                                                                 # the folder's photos count as B-roll you have
            cache = st.get("photos.json", {}) or {}
            have += PH.scenes(PH.describe(ai, PH.scan(Path(proj["folder"]), skipped), cache))
            st.put("photos.json", cache)
        except Exception as e:  # noqa: BLE001
            log.warning(f"looking at the photos failed: {e}")
        if any(f.get("library") for f in files):
            have += (st.get("library.json") or {}).get("photos") or []
        if not passages:
            st.put("stock.json", empty)
            return files, scenes
        talk = next((f for f in files if not f["broll"] and not f.get("library")), None)
        landscape = talk is None or talk["w"] >= talk["h"]
        lines = [f"P{n + 1} [{p[0].start:.0f}-{p[-1].end:.0f} s] " + " ".join(s.text for s in p) for n, p in enumerate(passages)]
        owned = "\n".join(f"- {str(s.get('what') or '')[:100]}" for s in have[:80]) or "(none: the video has no B-roll of its own)"
        prompt = ((f"The creator's notes: {brief[:1500]}\n\n" if brief else "") + f"B-roll the video already has:\n{owned}\n\n"
                  + "What is said, in passages:\n" + "\n".join(lines)[:20000]
                  + "\n\nFor each passage (by its number): is it covered, and if not, the search.")
        progress(0.0, "Finding what your footage doesn't show")
        try:
            res = ai.complete_json("broll_queries", STOCK_SYSTEM, prompt, STOCK_SCHEMA)
        except Exception as e:  # noqa: BLE001 - stock is a bonus
            log.warning(f"writing the stock searches failed: {e}")
            st.put("stock.json", empty)
            return files, scenes
        looks: list[tuple[str, int]] = []
        for x in res.data.get("passages") or []:
            n = int(x.get("n") or 0) - 1
            if not 0 <= n < len(passages) or (x.get("covered") and have):           # (with no B-roll at all nothing is covered)
                continue
            for lk in (x.get("looks") or [])[:1]:
                lk = " ".join(str(lk).split())[:80]
                if lk and lk.lower() not in {q.lower() for q, _ in looks}:
                    looks.append((lk, n))
        looks = looks[:STOCK_QUERIES]
        if not looks:
            log.info("your own B-roll covers what is said: no stock needed")
            st.put("stock.json", empty)
            return files, scenes
        sdir = self.stock_dir(proj)
        lib, lib_scenes, photos, found, used = [], {}, [], [], set()
        t = STOCK_BASE
        for k, (q, n) in enumerate(looks):
            progress(k / len(looks), f"Searching stock B-roll: {q}")
            if getattr(self, "live_job", None):
                live.stage(self.store, self.live_job, f"Searching Pixabay and Pexels: {q}")
            hits = [x for x in SK.broll(q, landscape=landscape) if f"{x['provider']}:{x['id']}" not in used and x["score"] >= 0.3]
            pick = next((x for x in hits if x["kind"] == "video"), None) or next((x for x in hits if x["kind"] == "photo"), None)
            if pick is None:
                self._live("stock", query=q, found=0)
                continue
            used.add(f"{pick['provider']}:{pick['id']}")
            meta = {key: pick.get(key) for key in ("provider", "id", "kind", "url", "page", "license", "credit", "creator", "title")} | {"query": q}
            path = str(sdir / SK.local_name(pick))
            tags = [str(x).strip() for x in (pick.get("tags") or []) if str(x).strip()][:5]
            what = q + (f" ({', '.join(tags)})" if tags else "")
            if pick["kind"] == "photo":
                a = STOCK_PHOTO_BASE + 10.0 * len(photos)
                photos.append({"start": a, "end": a + PH.SCENE_S, "what": f"(still photo, stock) {what}", "quality": 6.0,
                               "file": f"Stock - {pick['provider'].title()} photo {pick['id']}", "photo": path, "library": True, "stock": meta})
            else:
                dur = max(3.0, float(pick.get("duration") or 10.0))
                name = f"Stock - {pick['provider'].title()} {pick['id']}"
                lib.append({"name": name, "path": path, "rel": Path(path).name, "size": 0, "mtime": 0, "dur": round(dur, 3), "start": round(t, 3),
                            "end": round(t + dur, 3), "w": int(pick.get("w") or (1920 if landscape else 1080)),
                            "h": int(pick.get("h") or (1080 if landscape else 1920)), "fps": 30.0, "audio": False, "tc": 0.0, "tc_df": False,
                            "broll": True, "library": True, "stock": meta})
                lib_scenes[name] = [{"start": round(t + 0.2, 2), "end": round(t + max(1.5, min(dur - 1.0, 12.0)), 2),
                                     "what": f"(stock footage) {what}", "quality": 6.0}]
                t += dur + 10.0
            found.append({"query": q, "passage": n + 1, "provider": pick["provider"], "id": pick["id"], "kind": pick["kind"], "page": pick.get("page")})
            self._live("stock", query=q, found=len(hits), provider=pick["provider"], media=pick["kind"], page=pick.get("page"))
        st.put("stock.json", {"files": lib, "scenes": lib_scenes, "photos": photos, "found": found})
        log.info(f"stock B-roll: {len(found)} of {len(looks)} search(es) found something ({len(lib)} videos, {len(photos)} photos)")
        if getattr(self, "live_job", None):
            live.stage(self.store, self.live_job, f"Stock B-roll offered for {len(found)} passage(s) your footage doesn't show")
        return files + lib, {**scenes, **lib_scenes}

    def fetch_stock(self, cutaways: list[dict], files: list[dict], st: State) -> list[str]:
        """Download the stock shots the edit really uses - only those, once (a re-render, or another project, finds them in the Stock folder) -
        with a .stock.json beside each (site, id, page, licence, credit, the search). A shot that can't be downloaded is taken out (the
        speaker stays on screen). -> the credit lines of the shots used."""
        sj = st.get("stock.json") or {}
        sphotos = {x["photo"]: x for x in sj.get("photos") or [] if x.get("photo") and x.get("stock")}
        sfiles = [f for f in files if f.get("stock")]
        if not sphotos and not sfiles:
            return []
        from .. import stock as SK
        from .polish import _fps

        def stock_of(c: dict):
            if c.get("photo"):
                x = sphotos.get(c["photo"])
                return (c["photo"], x["stock"], None) if x else None
            f = next((f for f in sfiles if f["start"] - 1e-3 <= float(c.get("src", -1)) < f["end"]), None)
            return (f["path"], f["stock"], f) if f else None
        need: dict[str, tuple[dict, dict | None]] = {}
        for c in cutaways:
            got = stock_of(c)
            if got:
                need.setdefault(got[0], (got[1], got[2]))
        bad: set[str] = set()
        credits: list[str] = []
        for path, (meta, f) in need.items():
            dest = Path(path)
            try:
                if not dest.exists():
                    if getattr(self, "live_job", None):
                        live.stage(self.store, self.live_job, f"Downloading stock B-roll: {meta.get('query')} ({str(meta.get('provider')).title()})")
                    got = SK.fetch(self.cfg, dict(meta), into=dest.parent)
                    if got != dest:
                        raise RuntimeError(f"it was saved as {got.name}")
                if f is not None:                                         # the real length, size and rate (the site's numbers are rounded)
                    info = ffmpeg.probe(dest)
                    vs = ffmpeg.video_stream(info)
                    if not vs:
                        raise RuntimeError("no picture in it")
                    f["dur"] = round(min(float(info.get("format", {}).get("duration") or 0), float(f["dur"]) + 9.0), 3)
                    f["end"] = round(f["start"] + f["dur"], 3)
                    f["w"], f["h"], f["fps"] = int(vs.get("width") or f["w"]), int(vs.get("height") or f["h"]), float(_fps(vs))
                    f["size"], f["mtime"] = dest.stat().st_size, int(dest.stat().st_mtime)
            except Exception as e:  # noqa: BLE001 - that cutaway is left out, the video is not
                log.warning(f"stock {meta.get('provider')} {meta.get('id')} could not be used ({type(e).__name__}): that cutaway stays on the speaker")
                bad.add(path)
                continue
            credits.append(f"{meta.get('credit')} - {meta.get('page')}" if meta.get("page") else str(meta.get("credit")))
        if bad:
            for c in cutaways:
                got = stock_of(c)
                if got and got[0] in bad:
                    c["length"] = 0.0
            cutaways[:] = [c for c in cutaways if c["length"] > 0.3]
        if credits and getattr(self, "live_job", None):
            live.stage(self.store, self.live_job, f"{len(credits)} stock shot(s) used (credits go in Notes/credits.txt)")
        return sorted(set(credits))

    def graphics_of(self, folder: Path, st: State, files: list[dict] | None = None) -> list[dict]:
        """The graphics in the footage folder (a logo, lower thirds, titles, see-through videos: foldergfx.scan), looked at once (graphics.json).
        files = the footage already scanned (never probed again for alpha)."""
        from . import foldergfx as FG
        cache = st.get("graphics.json", {}) or {}
        try:
            gfx = FG.scan(folder, skipped, cache, {str(f["path"]) for f in files or [] if f.get("path")})
        except Exception as e:  # noqa: BLE001 - the video is made without them
            log.warning(f"looking for your graphics failed: {e}")
            return []
        st.put("graphics.json", cache)
        return gfx

    def _prog(self, engine: Engine, job: str, status: str, stage: str, frac: float) -> None:
        try:                                                             # the bar never goes back (the ticker may have crept past this step's value)
            frac = max(float(frac), float(self.store.get(job).get("progress") or 0.0)) if frac < 0.999 else frac
        except Exception:  # noqa: BLE001
            pass
        engine._progress(job, status, stage, frac)
        self._pbase = (job, float(frac), time.time())

    # -- steps -------------------------------------------------------------------------------
    def make(self, pid: str, body: dict | None = None) -> None:
        body = body or {}
        proj, folder, st, ai, engine = self._setup(pid)
        job = proj["job"]
        P = lambda status, stage, frac: self._prog(engine, job, status, stage, frac)       # noqa: E731
        opts = self.store.get(job)["options"]
        if opts.get("keep_cut"):                                              # "Add B-roll to a video": that video stays exactly as it is
            return self.make_keep_cut(pid, proj, folder, st, ai, engine, opts, P)
        P("transcribing", "Scanning the folder", 0.02)
        self._music_choice(opts, st)
        files = scan(folder)
        if not files:
            raise RuntimeError("no videos in that folder")
        notes_ = self.projects.get(pid).get("notes") or ""
        counts = MO.quick_counts(folder, skipped)
        det = MO.detect(notes_, n_photos=counts["photos"], n_videos=len(files), audio_clips=sum(1 for f in files if f.get("audio")),
                        setting=opts.get("montage") or "auto")
        if det:                                                              # known without listening: a slideshow, a montage asked for, clips with no sound
            return self.make_montage(pid, proj, folder, st, ai, engine, opts, P, det, files)
        audios = scan_audio(folder, proj.get("extra_audio") or [])
        group = self.find_group(files, audios, st, lambda fr, m: P("transcribing", m, 0.03 + 0.02 * fr))
        if group:
            files = virtualize(files, group)
            audios = []                                                     # the recorder (if any) is part of the group already
            self.rate_sound(st)                                             # which camera sounds cleanest: that sound is transcribed AND heard
            group = (st.get("multicam.json") or {}).get("group") or group
        if audios:
            P("transcribing", f"Syncing {len(audios)} audio recording(s) to the video", 0.03)
            cache = st.get("sync.json", {})
            AS.sync_all(files, audios, cache, lambda fr, m: P("transcribing", m, 0.03 + 0.02 * fr))
            st.put("sync.json", cache)
            for f in files:
                r = (cache.get(f["name"]) or {}).get("result")
                if r:
                    rec = next((x for x in audios if x["path"] == r["audio"]), {})
                    f["clean"] = {"audio": r["audio"], "offset": r["offset"], "score": r["score"], "name": r["audio_name"],
                                  "tc": rec.get("tc", 0.0), "dur": rec.get("dur"), "ch": rec.get("ch", 2)}
        st.put("files.json", files)
        all_words = self.transcribe(files, ai, st, opts.get("transcriber", "parakeet"), lambda f, m: P("transcribing", m, 0.05 + 0.3 * f), group,
                                    refresh=True)
        det = MO.detect(self.projects.get(pid).get("notes") or "", n_words=len(all_words), footage_s=sum(float(f.get("dur") or 0) for f in files),
                        n_photos=0, n_videos=len(files), setting=opts.get("montage") or "auto")
        if det and not group:                                                # nobody is talking: cut to music instead (a multicam shoot is never one)
            return self.make_montage(pid, proj, folder, st, ai, engine, opts, P, det, files)
        outline = FL.steps(self.projects.get(pid).get("notes") or "")
        if outline:
            all_words, fixes = FL.correct_names(all_words, outline)
            if fixes:
                st.put("name_corrections.json", fixes)
                P("transcribing", f"Corrected {len(fixes)} name spelling(s) from your notes", 0.34)
        P("finding", "Telling the interviewer from the guest", 0.35)
        words = self.guest_words(all_words, ai, st)
        P("finding", "Looking at the B-roll", 0.36)
        scenes = self.describe(files, words, ai, st, lambda f, m: P("finding", m, 0.36 + 0.06 * f))
        brief = self.projects.brief(self.projects.get(pid))
        if opts.get("library", True) and self.library() is not None:
            P("finding", "Searching your footage library for B-roll", 0.42)
            try:
                files, scenes = self.add_library(files, words, scenes, ai, st, brief, lambda f, m: P("finding", m, 0.42))
                st.put("files.json", files)
            except Exception as e:  # noqa: BLE001 - library B-roll is a bonus: the video is made from its own footage instead
                log.warning(f"footage library B-roll left out: {e}")
        if opts.get("stock"):                                               # stock B-roll (Pixabay / Pexels) where your own doesn't show it
            P("finding", "Looking for stock B-roll for what your footage doesn't show", 0.43)
            try:
                files, scenes = self.add_stock(files, words, scenes, ai, st, brief, self.projects.get(pid), lambda f, m: P("finding", m, 0.43))
                st.put("files.json", files)
            except Exception as e:  # noqa: BLE001 - stock is a bonus
                log.warning(f"stock B-roll left out: {e}")
        sents = seg(words)
        plan = self.plan(files, words, sents, scenes, ai, st, opts, brief, pid, lambda f, m: P("finding", m, 0.42 + 0.12 * f))
        self._live_lines(plan, sents)
        st.put("plan_words.json", {"words": all_words, "spans": (st.get("roles.json") or {}).get("spans", [])})   # what this edit is made of
        if plan.get("script_map"):
            self.store.update(job, status="approval", stage="Script map ready — review the exact lines, then approve rendering", progress=0.54, error=None)
            return
        self.render(pid, plan, files, words, scenes, ai, engine, st, folder, note="first cut" if not plan.get("version") else "made again",
                    all_words=all_words)

    def make_keep_cut(self, pid: str, proj: dict, folder: Path, st: State, ai: LocalAI, engine: LocalEngine, opts: dict, P) -> None:
        """"Add B-roll to a video": one finished (or single-take) video is the whole programme, played once from start to end exactly as
        it was delivered - no takes chosen, no fillers or pauses cut, nothing reordered, no length target, no zooms. It is transcribed, and
        B-roll from the rest of the folder (every other video there is B-roll, talking or not) and from the footage library is cut in over
        what is said. Final Cut gets the video as ONE clip on the main storyline with the B-roll connected above it."""
        job = proj["job"]
        P("transcribing", "Reading your video", 0.02)
        self._music_choice(opts, st)
        files = scan(folder)
        aroll = str(opts.get("aroll") or "")
        a_file = next((f for f in files if f["name"] == aroll or Path(f["rel"]).name == aroll), None)
        if a_file is None:
            raise RuntimeError(f"the video to add B-roll to ({aroll or 'none chosen'}) isn't in the folder any more")
        for f in files:                                                     # everything that isn't the video is B-roll, talking or not
            f["broll"] = f is not a_file
        st.put("files.json", files)
        all_words = self.transcribe([a_file], ai, st, opts.get("transcriber", "parakeet"),
                                    lambda fr, m: P("transcribing", "Transcribing your video" + (" (Parakeet)" if "Parakeet" in m else ""), 0.05 + 0.3 * fr),
                                    refresh=True)
        if not all_words:
            log.info("the video has no words to match B-roll to: B-roll is placed by the twelve-second rule only")
        words = all_words                                                   # (one speaker's programme: nothing is left out of it)
        brief = self.projects.brief(self.projects.get(pid))
        scenes: dict = {}
        if any(f["broll"] for f in files):
            P("finding", "Looking at your B-roll", 0.36)
            scenes = self.describe(files, words, ai, st, lambda fr, m: P("finding", m.replace("Looking at", "Looking at B-roll:"), 0.36 + 0.06 * fr))
            scenes.pop(a_file["name"], None)                               # (a quiet video is described too: it is never its own B-roll)
        if opts.get("library", True) and self.library() is not None:
            P("finding", "Finding B-roll in your footage library", 0.42)
            try:
                files, scenes = self.add_library(files, words, scenes, ai, st, brief, lambda fr, m: P("finding", m, 0.42))
                st.put("files.json", files)
            except Exception as e:  # noqa: BLE001 - library B-roll is a bonus: the folder's own B-roll is used instead
                log.warning(f"footage library B-roll left out: {e}")
        if opts.get("stock"):
            P("finding", "Looking for stock B-roll for what your footage doesn't show", 0.43)
            try:
                files, scenes = self.add_stock(files, words, scenes, ai, st, brief, self.projects.get(pid), lambda fr, m: P("finding", m, 0.43))
                st.put("files.json", files)
            except Exception as e:  # noqa: BLE001 - stock is a bonus
                log.warning(f"stock B-roll left out: {e}")
        sents = seg(words)
        notes = self.projects.get(pid).get("notes") or ""
        style = video_kind(ai, notes, opts.get("kind")) if notes.strip() or opts.get("kind") in KINDS else "story"
        if style == "ad":
            style = "promo"                                                 # (an ad's end card and cut-down rules would change the video)
        pool = [{**s, "file": f["name"]} for f in files if f["broll"] and not f.get("library") for s in scenes.get(f["name"], [])]
        try:                                                                 # photos in the folder: offered as B-roll like in a normal edit
            cache = st.get("photos.json", {}) or {}
            found = PH.describe(ai, PH.scan(folder, skipped), cache)
            st.put("photos.json", cache)
            pool = pool + PH.scenes(found)
        except Exception as e:  # noqa: BLE001 - photos are a bonus
            log.warning(f"looking at the photos failed: {e}")
        lib_names = {f["name"] for f in files if f.get("library")}
        pool = pool + [{**s, "file": k} for k, v in scenes.items() if k in lib_names for s in v]
        if any(f.get("library") and not f.get("stock") for f in files):
            pool = pool + ((st.get("library.json") or {}).get("photos") or [])
        if opts.get("stock"):
            pool = pool + ((st.get("stock.json") or {}).get("photos") or [])
        old = st.get("plan.json", {}) or {}
        title = " ".join(str(proj.get("title") or "").split()) or Path(a_file["name"]).stem
        if title.lower() in {Path(a_file["name"]).stem.lower(), Path(str(proj.get("folder") or "")).name.lower()}:     # (still the file's or folder's name: name it from what is said)
            earlier = " ".join(str(old.get("title") or "").split())
            title = earlier if earlier and earlier.lower() != title.lower() else (FL.name_it(ai, sents, notes) or title)
            if title.lower() != Path(a_file["name"]).stem.lower():
                self.projects.update(pid, title=title)                      # (the sidebar shows the project's name)
        plan = {"style": style, "keep_cut": {"file": a_file["name"], "start": a_file["start"], "end": a_file["end"]},
                "items": [{"kind": "visual", "start": a_file["start"], "end": a_file["end"]}], "pool": pool, "title": title,
                "summary": " ".join(s.text for s in sents[:6])[:400], "gap": 0.0, "hook_text": old.get("hook_text", ""),
                "music": (self._song_music(opts["song"]) if opts.get("song") else {"mode": "none"} if opts.get("music") in (None, "")
                          else {"mode": "auto"} if opts.get("music") == "auto" else {"mode": "auto", "wanted": str(opts.get("music"))}),
                "beat_sync": False, "target_s": None,
                "version": int(old.get("version", 0)), "history": old.get("history", []), "broll_notes": old.get("broll_notes", []),
                "broll_per_s": min(10.0, float(PROFILES.get(style, PROFILES["story"])["cut_per_s"]))}   # B-roll is the point here: about one every 10 s
        if opts.get("graphics") and not plan["hook_text"]:                  # a title card only when graphics were asked for
            plan["hook_text"] = self.hook(ai, {**plan, "items": [{"kind": "talk", "a": s.idx, "b": s.idx} for s in sents[:2]]}, sents, brief)
        plan["pending_render"], plan["pending_note"] = True, "B-roll added" if not plan.get("version") else "B-roll placed again"
        st.put("plan_words.json", {"words": all_words, "spans": []})
        st.put("plan.json", plan)
        if getattr(self, "live_job", None):
            live.stage(self.store, self.live_job, f"{a_file['name']} stays exactly as it is: {len(pool)} B-roll shot(s) to choose from")
        self.render(pid, plan, files, words, scenes, ai, engine, st, folder, note=plan["pending_note"], all_words=all_words)

    # -- montages and slideshows (montage.py): footage with no one speaking, cut to music; the plan is checked before anything renders ---------
    def montage_track(self, plan: dict, folder: Path, ai, st: State, title: str, notes: str, est_s: float):
        """The song, chosen FIRST (the cuts follow it): the person's own, one in the project's Music folder, else the best match in the music library
        for the mood the notes and the pace ask for. Returns a Track or None."""
        m = plan.get("music") or {"mode": "auto"}
        mt = plan["montage"]
        if m.get("mode") == "none":
            return None
        if m.get("song") and m.get("file"):
            t = MU.song(m["file"])
            if t:
                return t
        used = set(m.get("used", []))
        own = MU.own_tracks(folder / "Music")
        if own and not m.get("wanted"):
            return own[int(hashlib.sha1(title.encode()).hexdigest()[:6], 16) % len(own)]
        pace = MO.pace_key(mt.get("pace"))
        mood = next((w for w in MU.MOODS if re.search(rf"\b{w}\b", notes, re.I)), "") or \
            {"fast": "upbeat", "slow": "emotional", "auto": "chill" if mt["style"] == "slideshow" else "upbeat"}[pace]
        flavour = ("gentle warm instrumental slideshow" if mt["style"] == "slideshow" and pace != "fast" else
                   "energetic upbeat high energy" if pace == "fast" else "feel good upbeat") + " " + re.sub(r"\s+", " ", notes)[:120]
        t = MU.pick(self.cfg, mood=mood, genre=flavour, seed=title + str(len(used)), exclude=used, length_s=est_s)
        plan["music"] = {**m, "mode": "auto", "mood": mood, "genre": "", "why": f"{mood}, for {'a slideshow' if mt['style'] == 'slideshow' else 'a montage'}"}
        return t

    def _montage_canvas(self, files: list[dict], photos: list[dict]) -> tuple[int, int]:
        """1920x1080, or 1080x1920 when most of what is shown is upright."""
        tall = sum(1 for f in files if not f.get("photo") and f.get("h", 0) > f.get("w", 0))
        wide = sum(1 for f in files if not f.get("photo") and f.get("h", 0) <= f.get("w", 0))
        for ph in photos[:80]:
            sz = PH._size(ph["path"])
            if sz:
                tall, wide = (tall + 1, wide) if sz[1] > sz[0] else (tall, wide + 1)
        return (1080, 1920) if tall > wide else (1920, 1080)

    def _montage_thumbs(self, job: str, m: dict, files: list[dict]) -> None:
        """A small picture of every shot in the plan card (job folder: mt_<id>.jpg). A shot that can't be read just has none."""
        jobd = self.store.dir(job)
        by_name = {f["name"]: f for f in files}
        for c in m["cands"]:
            out = jobd / f"mt_{c['id']}.jpg"
            if out.exists():
                continue
            try:
                if c["kind"] == "photo":
                    b = PH._jpeg(c["path"], 240)
                    if b:
                        out.write_bytes(b)
                else:
                    f = by_name.get(c["name"])
                    if not f:
                        continue
                    ss = max(0.0, (float(c["start"]) + float(c["end"])) / 2.0 - f["start"])
                    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{ss:.2f}", "-i", f["path"], "-frames:v", "1",
                                        "-vf", "scale=240:-2" + self._lut_vf(f["path"]), "-q:v", "6", str(out)], capture_output=True, timeout=60)
                    if r.returncode:
                        out.unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001 - a plan card without one picture is fine
                log.info(f"montage picture for {c['id']}: {e}")

    def make_montage(self, pid: str, proj: dict, folder: Path, st: State, ai: LocalAI, engine: LocalEngine, opts: dict, P, det: dict, files: list[dict]) -> None:
        """Footage with no A-roll (a camp, an event, a folder of photos): the music is chosen first, the best shots are picked and put in order,
        each held for the pace the person chose and cut on the beats, and the plan stops for the person to check (the plan card) before it renders."""
        job = proj["job"]
        notes = self.projects.get(pid).get("notes") or ""
        keep_sound = MO.wants_sound(notes) and det["style"] != "slideshow"
        P("finding", "Looking at your photos and clips", 0.06)
        for f in files:                                                     # every clip is the programme itself; its own sound stays out unless asked
            f["broll"] = False
            f["audio_orig"] = f.get("audio_orig", f["audio"])
            f["audio"] = bool(f["audio_orig"]) and keep_sound
        st.put("files.json", files)
        scenes = self.describe(files, [], ai, st, lambda fr, m: P("finding", m, 0.08 + 0.2 * fr)) if files else {}
        raw = MO.scan_photos(folder, skipped)
        cache = st.get("photos.json", {}) or {}
        got: dict[str, dict] = {}
        for k in range(0, len(raw), 40):
            P("finding", f"Looking at the photos ({min(len(raw), k + 40)} of {len(raw)})", 0.28 + 0.14 * k / max(1, len(raw)))
            got.update({q["path"]: q for q in PH.describe(ai, raw[k:k + 40], cache)})
            st.put("photos.json", cache)
        photos = [got.get(q["path"]) or {**q, "what": Path(q["name"]).stem.replace("_", " "), "quality": 6.0} for q in raw]   # (undescribed: still shown)
        if not files and not photos:
            raise RuntimeError("there are no clips or photos in that folder")
        cands = MO.candidates(files, scenes, photos, when=MO.shot_time)
        keep, left = MO.select(cands)
        if not keep:
            raise RuntimeError("none of the shots were good enough to use (they all looked blurry or unusable)")
        order = MO.arc_order(keep)
        pace = MO.pace_key(opts.get("pace"))
        minutes = float(opts.get("long_minutes") or 0)
        old = st.get("plan.json", {}) or {}
        title = " ".join(str(proj.get("title") or "").split()) or folder.name
        music = (self._song_music(opts["song"]) if opts.get("song") else {"mode": "none"} if str(opts.get("music")) in ("none", "False", "off")
                 else {"mode": "auto", "wanted": str(opts["music"])} if opts.get("music") not in (None, "", "auto", True) else {"mode": "auto"})
        W, H = self._montage_canvas(files, photos)
        mt = {"style": det["style"], "why": det["why"], "line": det["line"], "pace": pace, "target_s": minutes * 60.0 if minutes else None,
              "dissolves": MO.wants_dissolves(notes), "keep_sound": keep_sound, "cands": cands, "order": order, "removed": [], "left_out": left, "canvas": [W, H]}
        plan = {"style": "story", "title": title, "items": [], "pool": [], "montage": mt, "music": music, "beat_sync": True, "target_s": mt["target_s"],
                "summary": "", "gap": 0.0, "hook_text": "", "version": int(old.get("version", 0)), "history": old.get("history", []), "broll_notes": []}
        P("finding", "Choosing the music", 0.46)
        est = min(MO.MAX_AUTO_S, sum(sum(MO.PACE[pace]["photo" if c["kind"] == "photo" else "clip"]) / 2.0 for c in keep))
        track = self.montage_track(plan, folder, ai, st, title, notes, mt["target_s"] or est)
        if track:
            grid = self.beat_grid(track, st)
            plan["music"] = {**plan["music"], "file": str(track.path), "title": track.title, "used": sorted(set(plan["music"].get("used", [])) | {track.path.name})}
            mt["song"] = {"title": track.title, "file": str(track.path), "bpm": (grid or {}).get("bpm"), "why": plan["music"].get("why", ""), "mood": plan["music"].get("mood", "")}
            mt["period"] = (grid or {}).get("period")
            mt["music_start"] = max(0.0, (grid or {}).get("first", 0.0)) if grid else 0.0
        else:
            plan["music"] = {"mode": "none"}
            mt["song"], mt["period"], mt["music_start"] = {}, None, 0.0
        MO.layout(mt)
        P("finding", "Writing the plan", 0.52)
        mt["summary"], new_title = self._montage_story(ai, mt, notes, title)
        if new_title and title.lower() in {folder.name.lower(), ""} or (new_title and not proj.get("title")):
            plan["title"] = new_title
            self.projects.update(pid, title=new_title)
        plan["items"] = mt["items"]
        self._montage_thumbs(job, mt, files)
        plan["awaiting_approval"], plan["pending_render"], plan["pending_note"] = True, True, f"{det['style']} planned"
        st.put("plan_words.json", {"words": [], "spans": []})
        st.put("plan.json", plan)
        self.store.update(job, status="approval", stage="The plan is ready: check it, then make the video", progress=0.55, error=None)

    def _montage_story(self, ai, mt: dict, notes: str, title: str) -> tuple[str, str]:
        """(a sentence or two on how the video unfolds, a title): from the shots in play order. The AI may fail: then it is built from the first, middle
        and last shots."""
        cands = {c["id"]: c for c in mt["cands"]}
        seq = [cands[i["id"]]["what"] for i in mt["items"]]
        if not seq:
            return "", ""
        step = max(1, len(seq) // 24)
        try:
            res = ai.complete_json("montage_story", "You describe a finished video's story for the person who asked for it: plain, concrete, no hype.",
                                   f"What the person asked for: {notes or '(nothing written)'}\nThe shots, in play order: " + " | ".join(seq[::step])[:3500]
                                   + "\n\nWrite a title (3-7 words) and how the video unfolds in one or two sentences that start with 'It opens on'.",
                                   {"type": "object", "properties": {"title": {"type": "string"}, "summary": {"type": "string"}}, "required": ["title", "summary"]})
            summ = " ".join(str(res.data.get("summary") or "").split())[:400]
            ttl = " ".join(str(res.data.get("title") or "").replace('"', "").split())[:90]
            if summ:
                return summ, ttl
        except Exception as e:  # noqa: BLE001 - built from the shots instead
            log.info(f"montage story: {e}")
        return MO.default_summary(mt), ""

    def montage_card(self, pid: str) -> dict | None:
        """The plan card of a montage project (None for any other)."""
        proj = self.projects.get(pid)
        plan = State(home_of(proj)).get("plan.json") or {}
        mt = plan.get("montage")
        if not mt:
            return None
        return MO.card(mt, title=plan.get("title") or "", notes=proj.get("notes") or "", job=proj["job"])

    def montage_edit(self, pid: str, op: dict) -> dict:
        """One click on the plan card: remove / put back / move a shot, change the pace, the song, the length or the dissolves. The plan is laid out
        again (nothing renders). Returns the new card."""
        with self.review_lock:
            proj, folder, st, ai, engine = self._setup(pid)
            plan = st.get("plan.json") or {}
            mt = plan.get("montage")
            if not mt or not plan.get("awaiting_approval"):
                raise ValueError("there is no montage plan waiting for you")
            ids = {c["id"] for c in mt["cands"]}
            kind, cid = str(op.get("op") or ""), str(op.get("id") or "")
            if kind == "remove" and cid in mt["order"]:
                mt["removed"] = [*mt.get("removed", []), cid]
            elif kind == "restore" and cid in ids:
                mt["removed"] = [x for x in mt.get("removed", []) if x != cid]
                if cid not in mt["order"]:                                  # (a shot the planner left out: it goes at the end of its time)
                    mt["order"].append(cid)
                    cands = {c["id"]: c for c in mt["cands"]}
                    mt["order"].sort(key=lambda i: tuple(cands[i]["t"]))
                    mt["left_out"] = [g for g in mt.get("left_out", []) if g["id"] != cid]
            elif kind == "move" and cid in mt["order"]:
                before = str(op.get("before") or "")
                mt["order"].remove(cid)
                mt["order"].insert(mt["order"].index(before) if before in mt["order"] else len(mt["order"]), cid)
            elif kind == "pace":
                mt["pace"] = MO.pace_key(op.get("value"))
                self.store.update(proj["job"], options={**self.store.get(proj["job"])["options"], "pace": {"auto": "natural"}.get(mt["pace"], mt["pace"])})
            elif kind == "length":
                minutes = max(0.0, min(600.0, float(op.get("minutes") or 0)))
                mt["target_s"] = minutes * 60.0 if minutes else None
                plan["target_s"] = mt["target_s"]
            elif kind == "dissolves":
                mt["dissolves"] = bool(op.get("value"))
            elif kind == "song":
                notes = proj.get("notes") or ""
                want = str(op.get("value") or "")
                if want and want != "another":
                    if MU.song(want) is None:
                        raise ValueError("that song isn't an audio file on this Mac")
                    plan["music"] = self._song_music(want, plan.get("music"))
                else:
                    plan["music"] = {k: v for k, v in (plan.get("music") or {}).items() if k not in ("file", "song", "title")} or {"mode": "auto"}
                    plan["music"]["mode"] = "auto"
                track = self.montage_track(plan, folder, ai, st, plan.get("title") or "", notes, mt.get("total_s") or 120.0)
                if track:
                    grid = self.beat_grid(track, st)
                    plan["music"] = {**plan["music"], "file": str(track.path), "title": track.title, "used": sorted(set(plan["music"].get("used", [])) | {track.path.name})}
                    mt["song"] = {"title": track.title, "file": str(track.path), "bpm": (grid or {}).get("bpm"), "why": plan["music"].get("why", ""), "mood": plan["music"].get("mood", "")}
                    mt["period"] = (grid or {}).get("period")
                    mt["music_start"] = max(0.0, (grid or {}).get("first", 0.0)) if grid else 0.0
            else:
                raise ValueError("that change isn't one the plan card knows")
            MO.layout(mt)
            mt["summary"] = MO.default_summary(mt)
            plan["items"] = mt["items"]
            plan["montage"] = mt
            st.put("plan.json", plan)
            return MO.card(mt, title=plan.get("title") or "", notes=proj.get("notes") or "", job=proj["job"])

    def bake_montage(self, plan: dict, st: State) -> list[dict]:
        """The photos of a montage as short animated clips (a gentle move each, held as long as the plan says); they are shots of the programme."""
        mt = plan["montage"]
        W, H = mt.get("canvas") or (1920, 1080)
        pdir = st.dir / "photos"
        pdir.mkdir(exist_ok=True)
        out, k, gone = [], 0, set()
        for it in plan["items"]:
            if "photo" not in it:
                continue
            effect = MO.EFFECTS[k % len(MO.EFFECTS)]
            k += 1
            secs = float(it["end"]) - float(it["start"]) + 0.6
            mov = pdir / f"{safe(Path(it['photo']).stem, 40)}_{PH.key(it['photo'], effect, secs, W, H)[:8]}.mp4"      # (Final Cut shows this name)
            try:
                if not mov.exists():
                    PH.bake(it["photo"], effect, secs, W, H, 30.0, mov)
            except Exception as e:  # noqa: BLE001 - that photo is left out
                log.warning(f"animating {Path(it['photo']).name} failed: {e}")
                gone.add(it["id"])
                continue
            out.append({"name": f"photo {it['id']} {Path(it['photo']).name}", "path": str(mov), "start": float(it["start"]), "end": float(it["start"]) + secs, "dur": secs,
                        "w": W, "h": H, "fps": 30.0, "audio": False, "broll": False, "photo": True, "mphoto": True, "tc": 0.0})
        if gone:
            plan["items"] = [it for it in plan["items"] if it.get("id") not in gone]
        return out

    def rerender(self, pid: str) -> None:
        """Render the saved edit again - nothing is re-planned (after a failed render, or to pick up a fix): the same cut, no new AI choices
        about what to keep."""
        proj, folder, st, ai, engine = self._setup(pid)
        plan = st.get("plan.json")
        if plan and plan.get("montage") and plan.get("items"):                  # a montage: its own words are none; the shots are in the plan
            files = [f for f in (st.get("files.json") or []) if not f.get("mphoto")]
            return self.render(pid, plan, files, [], {}, ai, engine, st, folder, note=plan.get("pending_note") or "rendered again", all_words=[])
        if not plan or not plan.get("items"):
            raise RuntimeError("there is no saved edit to render yet - press Make the video")
        files, group, all_words, words = self.saved_inputs(folder, st, ai, self.store.get(proj["job"])["options"].get("transcriber", "parakeet"))
        scenes = self.describe(files, words, ai, st, lambda f, m: None)
        self.render(pid, plan, files, words, scenes, ai, engine, st, folder, note=plan.get("pending_note") or "rendered again", all_words=all_words)

    def music(self, pid: str, song: str = "") -> None:
        """A finished video gets a new song (or no music, with song=""): the same edit rendered again - nothing re-cut, no B-roll moved -
        with that song under it, ducked under the speech; the Final Cut project gets it as its own clip. The project keeps the song for
        later changes and re-renders."""
        proj, folder, st, ai, engine = self._setup(pid)
        plan = st.get("plan.json")
        if not plan or not plan.get("items"):
            raise RuntimeError("there is no finished edit yet - make the video first")
        song = str(song or "").strip()
        if song and MU.song(song) is None:
            raise ValueError("that song isn't an audio file on this Mac")
        job = proj["job"]
        opts = dict(self.store.get(job)["options"])
        if song:
            opts["song"], opts["music"] = song, opts.get("music") or "auto"
            plan["music"] = self._song_music(song, plan.get("music"))
            note = f"new song: {Path(song).stem}"
        else:
            opts.pop("song", None)
            opts["music"] = ""
            plan["music"] = {"mode": "none"}
            note = "music taken out"
        if plan.get("beat_sync"):                                            # (snapping to a new beat would move the cuts: not here)
            plan["beat_sync"] = False
        self.store.update(job, options=opts)
        st.put("plan.json", {**plan, "pending_note": note})
        if getattr(self, "live_job", None):
            live.stage(self.store, self.live_job, ("Song: " + Path(song).stem) if song else "Taking the music out")
        self.rerender(pid)

    # -- one B-roll shot of a finished video swapped (the strip under the player) ------------------------------------------------------
    # The placed B-roll of the last render is kept in render_cache.json (final.cutaways); a re-render reuses it exactly while the cut and
    # the B-roll pool are the same. So swapping one entry there and rendering again changes that shot and nothing else.
    def _shot_state(self, pid: str) -> dict:
        """What the strip needs, without starting any AI: {proj, job, st, plan, rc, placed, pool, files}."""
        proj = self.projects.get(pid)
        st = State(home_of(proj))
        plan = st.get("plan.json") or {}
        if not plan.get("items"):
            raise ValueError("there is no finished edit yet - make the video first")
        rc = st.get("render_cache.json") or {}
        placed = (rc.get("final") or {}).get("cutaways")
        if placed is None or plan.get("pool") is None:
            raise ValueError("render the video once more first: it was made before shots could be swapped")
        pool = [s for s in plan["pool"] if s["quality"] >= (6 if plan.get("style") == "ad" else 4)]   # (the same pool render() numbers its shots in)
        files = list(st.get("files.json") or [])
        return {"proj": proj, "job": proj["job"], "st": st, "plan": plan, "rc": rc, "placed": placed, "pool": pool, "files": files}

    def _shot_source(self, s: dict, sc: dict, src: float | None = None, length: float = 2.0):
        """(path, seconds into it, is_photo) of a pool shot, or None when a re-render couldn't cut it (its file isn't in this project)."""
        if sc.get("photo"):
            return (sc["photo"], 0.0, True) if Path(sc["photo"]).is_file() else None
        f = next((f for f in s["files"] if f.get("name") == sc.get("file") and f.get("path")), None)
        if f is None or not Path(f["path"]).is_file():
            return None
        t = (sc["start"] if src is None else src) - f["start"] + min(0.6, max(0.0, length / 2))
        return f["path"], max(0.0, t), False

    def _shot_thumb(self, job: str, source) -> str:
        """A small picture of a shot in the job folder (shot_<hash>.jpg, served at /media/<job>/...): "" when it couldn't be made."""
        if not source:
            return ""
        path, t, photo = source
        try:
            stt = Path(path).stat()
        except OSError:
            return ""
        name = "shot_" + hashlib.sha1(f"{path}:{stt.st_size}:{int(stt.st_mtime)}:{t:.2f}".encode()).hexdigest()[:12] + ".jpg"
        out = self.store.dir(job) / name
        if not out.exists():
            try:
                if photo:
                    from PIL import Image, ImageOps
                    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                    im.thumbnail((320, 320))
                    im.save(out, "JPEG", quality=80)
                else:
                    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                                    "-vf", "scale=320:-2", "-q:v", "5", str(out)], capture_output=True, timeout=60)
            except Exception as e:  # noqa: BLE001 - a missing thumbnail is not an error
                log.warning(f"no thumbnail for {Path(path).name}: {e}")
        return name if out.exists() else ""

    def shots(self, pid: str) -> dict:
        """The finished video's B-roll in time order: [{k, at, length, what, photo, thumb}] (k = its place in render_cache)."""
        s = self._shot_state(pid)
        out = []
        pend = [p for p in (s["plan"].get("pending_changes") or []) if isinstance(p, dict) and p.get("at") is not None]
        for k, c in enumerate(s["placed"]):
            sc = s["pool"][c["scene"]] if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(s["pool"]) else {}
            if not sc or float(c.get("length") or 0) <= 0:
                continue
            out.append({"k": k, "at": float(c["at"]), "length": float(c["length"]), "what": str(c.get("what") or sc.get("what") or ""),
                        "photo": bool(sc.get("photo")), "thumb": self._shot_thumb(s["job"], self._shot_source(s, sc, c.get("src"), float(c["length"]))),
                        "changed": any(abs(float(p["at"]) - float(c["at"])) < 0.06 and p.get("kind") != "removed" for p in pend)})   # (not rendered yet)
        return {"job": s["job"], "shots": sorted(out, key=lambda x: x["at"]), "pending": len(s["plan"].get("pending_changes") or [])}

    def _spoken_near(self, s: dict, a: float, b: float) -> str:
        """The words heard in the finished video between a and b seconds (the cut's intervals map the shoot's words onto it)."""
        try:
            iv = json.loads((self.store.dir(s["job"]) / "plan.json").read_text()).get("intervals") or []
        except (OSError, ValueError):
            return ""
        words = (s["st"].get("plan_words.json") or {}).get("words") or []
        if not iv or not words:
            return ""
        return " ".join(w["text"] for w in remap_words(words, [tuple(x) for x in iv], 1.0) if a <= float(w["start"]) <= b)

    def shot_choices(self, pid: str, k: int, n: int = 8) -> dict:
        """Other shots that could go where shot k is: unused pool shots this project can cut, ranked by how well their description (and,
        when the footage catalogue's picture model is already loaded, how they look) fits what is said there. No AI call."""
        s = self._shot_state(pid)
        if not 0 <= k < len(s["placed"]):
            raise ValueError("that B-roll shot isn't in the video any more")
        c = s["placed"][k]
        used = {x.get("scene") for x in s["placed"]}
        spoken = self._spoken_near(s, float(c["at"]) - 2.0, float(c["at"]) + float(c["length"]) + 2.0)
        stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "it", "that", "this", "with", "for", "we", "i",
                "you", "they", "be", "so", "but", "just", "like", "about", "what", "there", "their", "our", "my", "your", "have", "has", "do"}
        toks = lambda t: {w for w in re.findall(r"[a-z]{3,}", (t or "").lower()) if w not in stop}       # noqa: E731
        said = toks(spoken)
        look: dict[str, list[tuple[float, float, float]]] = {}
        cat = self.catalog
        if spoken and cat is not None and getattr(cat, "_embedder", None) is not None:   # (only when the picture model is loaded already)
            try:
                for r in cat.search(spoken[:300], n=120, per_file=3):
                    look.setdefault(str(r.get("path")), []).append((float(r.get("start") or 0), float(r.get("end") or 0), float(r.get("score") or 0)))
            except Exception as e:  # noqa: BLE001 - the words alone still rank them
                log.warning(f"picture search for the B-roll choices failed: {e}")
        ranked = []
        ruled_out = AV.bad_pool(AV.load(s["st"]), s["pool"], s["files"])        # ("Don't use this": never offered again)
        for i, sc in enumerate(s["pool"]):
            if i in used or i in ruled_out:
                continue
            src = self._shot_source(s, sc)
            if src is None:
                continue
            w = toks(sc.get("what"))
            score = len(w & said) / (1.0 + len(w) ** 0.5) + 0.02 * float(sc.get("quality") or 0)
            f = next((f for f in s["files"] if f.get("name") == sc.get("file")), None)
            if f and f.get("path") in look:
                a, b = sc["start"] - f["start"], sc["end"] - f["start"]
                score += 4.0 * max([sco for x0, x1, sco in look[f["path"]] if x0 < b and x1 > a] or [0.0])
            ranked.append((score, i, sc, src))
        ranked.sort(key=lambda x: -x[0])
        out = [{"choice": i, "what": str(sc.get("what") or ""), "photo": bool(sc.get("photo")),
                "length": round(float(c["length"]) if sc.get("photo") else min(float(c["length"]), float(sc["end"]) - float(sc["start"])), 1),
                "thumb": self._shot_thumb(s["job"], src)} for _, i, sc, src in ranked[:n]]
        return {"k": k, "at": float(c["at"]), "length": float(c["length"]), "what": str(c.get("what") or ""), "spoken": spoken[:400], "choices": out}

    def shot_search(self, pid: str, k: int, q: str, n: int = 12) -> dict:
        """The Replace sheet's search box: unused shots of this project whose description has the words, then shots from the footage
        library that look like them (Catalog.search, the same as Find footage). Library choices are "lib:<id>:<start>:<end>": picking
        one adds that file to the project (replace_shot)."""
        s = self._shot_state(pid)
        if not 0 <= k < len(s["placed"]):
            raise ValueError("that B-roll shot isn't in the video any more")
        c = s["placed"][k]
        q = " ".join(str(q or "").split())[:200]
        if not q:
            return {"k": k, "q": "", "results": []}
        ln = float(c["length"])
        want = {w for w in re.findall(r"[a-z]{3,}", q.lower())}
        used = {x.get("scene") for x in s["placed"]}
        mine = []
        avoid = AV.load(s["st"])
        ruled_out = AV.bad_pool(avoid, s["pool"], s["files"])
        for i, sc in enumerate(s["pool"]):
            text = str(sc.get("what") or "").lower()
            hit = len(want & set(re.findall(r"[a-z]{3,}", text))) + (2 if q.lower() in text else 0)
            if i in used or i in ruled_out or not hit:
                continue
            src = self._shot_source(s, sc)
            if src is None:
                continue
            mine.append((hit, {"choice": i, "what": str(sc.get("what") or ""), "photo": bool(sc.get("photo")), "where": "project",
                               "length": round(ln if sc.get("photo") else min(ln, float(sc["end"]) - float(sc["start"])), 1),
                               "pic": f"/media/{s['job']}/{self._shot_thumb(s['job'], src)}" if src else ""}))
        mine.sort(key=lambda x: -x[0])
        lib = []
        cat = self.library()
        if cat is not None:
            here = {}                                                       # what this project already has from each file: [(a, b)]
            byname = {f.get("name"): f for f in s["files"]}
            for sc in s["pool"]:
                f = byname.get(sc.get("file"))
                if f and f.get("path"):
                    here.setdefault(f["path"], []).append((float(sc["start"]) - float(f["start"]), float(sc["end"]) - float(f["start"])))
            try:
                found = cat.search(q, n=n * 2, per_file=1)
            except Exception as e:  # noqa: BLE001 - the project's own shots still show
                log.warning(f"library search for a B-roll swap failed: {e}")
                found = []
            for r in found:
                a, b = float(r.get("start") or 0), float(r.get("end") or 0)
                if any(x0 < b and x1 > a for x0, x1 in here.get(str(r.get("path")), [])):
                    continue                                                # (already one of this project's shots)
                if AV.is_avoided(avoid, f"Library - {r.get('rel')}", a, b, r.get("kind") == "photo"):
                    continue                                                # (ruled out with "Don't use this")
                photo = r.get("kind") == "photo"
                lib.append({"choice": f"lib:{r['id']}:{a:.2f}:{b:.2f}", "what": str(r.get("what") or Path(str(r.get("rel") or "")).stem),
                            "photo": photo, "where": "library", "folder": str(Path(str(r.get("rel") or "")).parent),
                            "length": round(ln if photo else min(ln, max(0.0, b - a)), 1),
                            "pic": f"/api/library/thumb/{r['id']}/{int(r.get('k') or 0)}"})
        return {"k": k, "q": q, "results": [x for _, x in mine[:6]] + lib[:n]}

    def _lib_choice(self, s: dict, choice: str) -> tuple[dict, dict | None]:
        """A library choice ("lib:<id>:<start>:<end>") -> (the pool scene for it, the file entry to add to files.json or None), the way
        add_library adds catalog footage: a "Library - <rel>" file on the timeline past LIB_BASE (photos: a photo scene)."""
        m = re.fullmatch(r"lib:([0-9A-Za-z_-]{4,80}):(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", str(choice))
        if not m:
            raise ValueError("pick one of the shots")
        cat = self.library()
        rec = cat.load(m.group(1)) if cat is not None else None
        if not rec or not Path(str(rec.get("path") or "")).is_file():
            raise ValueError("that shot's file isn't there any more (is its drive plugged in?)")
        a, b = float(m.group(2)), float(m.group(3))
        dur = float(rec.get("dur") or 0)
        mid = (a + b) / 2
        sc0 = next((x for x in (rec.get("scenes") or []) if x["start"] - 0.5 <= mid <= x["end"] + 0.5), None) or ((rec.get("scenes") or [None])[0]) or {}
        what = str(sc0.get("what") or Path(rec["rel"]).stem)
        name = f"Library - {rec['rel']}"
        if rec.get("kind") == "photo":
            start = max([LIB_PHOTO_BASE - 10.0] + [float(x["start"]) for x in s["plan"]["pool"] if x.get("photo") and x.get("library")]) + 10.0
            return {"start": start, "end": start + PH.SCENE_S, "what": f"(still photo) {what}", "quality": 8.0, "file": name,
                    "photo": rec["path"], "library": True}, None
        if not (0.0 <= a < b and (not dur or b <= dur + 0.5)):
            raise ValueError("pick one of the shots")
        f = next((x for x in s["files"] if x.get("path") == rec["path"]), None)
        new = None
        if f is None:
            t0 = max([LIB_BASE] + [float(x["end"]) + 10.0 for x in s["files"] if float(x.get("start") or 0) >= LIB_BASE and not x.get("photo")])
            f = new = {"name": name, "path": rec["path"], "rel": rec["rel"], "size": rec.get("size"), "mtime": rec.get("mtime"),
                       "dur": dur, "start": round(t0, 3), "end": round(t0 + dur, 3), "w": int(rec.get("w") or 1920), "h": int(rec.get("h") or 1080),
                       "fps": float(rec.get("fps") or 30.0), "audio": bool(rec.get("audio")), "tc": float(rec.get("tc") or 0.0),
                       "tc_df": bool(rec.get("tc_df")), "broll": True, "library": True}
        return {"start": round(float(f["start"]) + a, 2), "end": round(float(f["start"]) + min(b, dur or b), 2), "what": what,
                "quality": 8.0, "file": f["name"], "library": True}, new

    def _swap(self, s: dict, k: int, choice) -> tuple[list[dict], str]:
        """The placed shots with shot k swapped for pool shot `choice` (or taken out: "speaker"), and a note for the page."""
        if not 0 <= k < len(s["placed"]):
            raise ValueError("that B-roll shot isn't in the video any more")
        placed = [dict(x) for x in s["placed"]]
        c = placed[k]
        if choice == "speaker":
            placed.pop(k)
            return placed, "B-roll shot taken out"
        try:
            i = int(choice)
        except (TypeError, ValueError):
            raise ValueError("pick one of the shots") from None
        if not 0 <= i < len(s["pool"]) or i in {x.get("scene") for j, x in enumerate(placed) if j != k}:
            raise ValueError("that shot isn't one of the choices any more")
        sc = s["pool"][i]
        if self._shot_source(s, sc) is None:
            raise ValueError("that shot's file isn't there any more (is its drive plugged in?)")
        ln = float(c["length"]) if sc.get("photo") else min(float(c["length"]), float(sc["end"]) - float(sc["start"]))
        placed[k] = {**c, "src": round(float(sc["start"]), 2), "scene": i, "what": sc.get("what", ""), "why": "chosen by you",
                     "file": sc.get("file"), "photo": sc.get("photo"), "length": round(ln, 2)}
        return placed, "B-roll shot swapped"

    def check_replace(self, pid: str, k: int, choice) -> None:
        """Raises ValueError (for the page, at once) when replace_shot would fail."""
        s = self._shot_state(pid)
        if isinstance(choice, str) and choice.startswith("lib:"):
            if not 0 <= k < len(s["placed"]):
                raise ValueError("that B-roll shot isn't in the video any more")
            self._lib_choice(s, choice)
            return
        self._swap(s, k, choice)

    def replace_shot(self, pid: str, k: int, choice, stage: bool = False):
        """Runner action: shot k becomes pool shot `choice` (or the speaker again: "speaker"), and the video renders again with every other
        cut, shot, song and graphic as it was. With stage=True (the page's Replace) nothing renders: the plan is only marked out of date
        (stage_replace_shot), and the person presses Re-render when they have made all their changes."""
        s = self._shot_state(pid)
        if stage:
            self._stage_begin(s)
        at0 = float(s["placed"][k]["at"]) if 0 <= k < len(s["placed"]) else None
        if isinstance(choice, str) and choice.startswith("lib:"):          # a shot from the footage library: it joins this project first
            sc, new = self._lib_choice(s, choice)
            same = next((i for i, x in enumerate(s["pool"]) if x.get("file") == sc.get("file") and abs(float(x["start"]) - sc["start"]) < 0.5), None)
            if same is None:
                # Appended at the END of the pool, so every placed shot keeps its number; pool_n remembers the pool the saved placement was
                # made with, so render() still reuses it although the pool grew (see its reuse rule).
                rc0 = s["rc"]
                n0 = rc0["pool_n"] if isinstance(rc0.get("pool_n"), int) else len(s["pool"])
                plan = {**s["plan"], "pool": list(s["plan"]["pool"]) + [sc]}
                files = s["files"] + ([new] if new else [])
                if new:
                    s["st"].put("files.json", files)
                s["st"].put("plan.json", plan)
                s = {**s, "plan": plan, "files": files, "rc": {**rc0, "pool_n": n0},
                     "pool": [x for x in plan["pool"] if x["quality"] >= (6 if plan.get("style") == "ad" else 4)]}
                same = len(s["pool"]) - 1
            choice = same
        placed, note = self._swap(s, k, choice)
        rc = s["rc"]
        s["st"].put("render_cache.json", {**rc, "final": {**rc["final"], "cutaways": placed}, "pinned": True})   # (an ad keeps it too)
        s["st"].put("plan.json", {**s["plan"], "pending_note": note})
        if stage:
            n = self._stage_end(s["st"], "removed" if choice == "speaker" else "replaced", note, at0)
            return {"ok": True, "staged": True, "pending": n, "message": note}
        live.stage(self.store, s["job"], note + (f": {placed[k]['what'][:80]}" if choice != "speaker" and k < len(placed) else ""))
        self.rerender(pid)

    def stage_replace_shot(self, pid: str, k: int, choice) -> dict:
        """Replace without rendering: the video stays as it was, marked out of date (pending_changes) until Re-render."""
        return self.replace_shot(pid, k, choice, stage=True)

    # -- changes made on the strip wait for "Re-render" ---------------------------------------------------------------------------------
    # 2026-11-06: "the second I clicked the B-roll shot, it started re-rendering. But I don't want it to do that if I've got more changes
    # to make." Replace, Edit (trim/slide) and "Don't use this" only change the saved placement and plan; plan.json gets pending_render and
    # a pending_changes list ({kind, note, at}); the finished video and its job are untouched (status stays done). Nothing starts a
    # render by itself: resume() only restarts jobs that were queued. A render of any kind (Re-render, a typed change, a transcript edit)
    # uses the placement as saved and clears the flags (render() pops them).
    def _stage_begin(self, s: dict) -> None:
        """Before the first waiting change: what "Discard changes" goes back to."""
        if s["plan"].get("pending_changes"):
            return
        s["st"].put("shots_baseline.json", {"plan": {k: v for k, v in s["plan"].items() if k not in ("pending_render", "pending_note", "pending_changes")},
                                            "render_cache": s["rc"], "files": s["files"]})

    def _stage_end(self, st: State, kind: str, note: str, at: float | None) -> int:
        """Mark the plan out of date and list the change (one entry per shot: the latest) -> how many changes are waiting."""
        plan = dict(st.get("plan.json") or {})
        ch = [c for c in (plan.get("pending_changes") or []) if isinstance(c, dict)]
        if at is not None:
            ch = [c for c in ch if c.get("at") is None or abs(float(c["at"]) - float(at)) >= 0.06]
        ch.append({"kind": kind, "note": note, "at": None if at is None else round(float(at), 2)})
        plan["pending_changes"] = ch
        plan["pending_render"] = True
        plan["pending_note"] = "; ".join(dict.fromkeys(str(c.get("note") or "") for c in ch if c.get("note")))[:300] or "B-roll changed"
        st.put("plan.json", plan)
        return len(ch)

    def discard_pending(self, pid: str) -> dict:
        """Back to the B-roll as it was in the last render: the placement, the pool and the files saved before the first waiting change."""
        if self.running == pid:
            raise ValueError("this video is being made right now")
        proj = self.projects.get(pid)
        st = State(home_of(proj))
        base = st.get("shots_baseline.json")
        plan = st.get("plan.json") or {}
        if not base or not plan.get("pending_changes"):
            raise ValueError("there are no waiting changes to discard")
        st.put("render_cache.json", base["render_cache"])
        st.put("files.json", base["files"])
        st.put("plan.json", base["plan"])
        return {"ok": True, "pending": 0}

    # -- trim / slide one B-roll shot (the Edit button on the strip) ---------------------------------------------------------------------
    MIN_SHOT_S = 0.5

    def _shot_file(self, s: dict, c: dict):
        """(pool scene, the project's file entry) of placed shot c, or raises for a photo / a file that is gone."""
        sc = s["pool"][c["scene"]] if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(s["pool"]) else None
        if not sc:
            raise ValueError("that B-roll shot isn't in the video any more")
        if sc.get("photo") or c.get("photo"):
            raise ValueError("a photo has no film to trim: use Replace to pick another picture")
        f = next((f for f in s["files"] if f.get("name") == sc.get("file") and f.get("path")), None)
        if f is None or not Path(f["path"]).is_file():
            raise ValueError("that shot's file isn't there any more (is its drive plugged in?)")
        return sc, f

    def _shot_limits(self, s: dict, k: int, c: dict) -> float:
        """How long shot k may be on the timeline: until the next shot starts, and not past the end of the finished video."""
        later = sorted(float(x["at"]) for j, x in enumerate(s["placed"]) if j != k and float(x["at"]) > float(c["at"]) + 1e-6)
        room = (later[0] if later else 1e9) - float(c["at"])
        try:
            clip = (self.store.get(s["job"]).get("clips") or [None])[0] or {}
            end = float(clip.get("duration") or 0)
        except (KeyError, OSError, TypeError, ValueError):
            end = 0.0
        if end > 0:
            room = min(room, end - float(c["at"]))
        return max(self.MIN_SHOT_S, room)

    def shot_edit_info(self, pid: str, k: int) -> dict:
        """What the trim editor needs for shot k: the source file, its in/out (seconds into that file), its frame rate, how long it may get
        on the timeline, and a strip of stills around it (made once into the job folder)."""
        s = self._shot_state(pid)
        if not 0 <= k < len(s["placed"]):
            raise ValueError("that B-roll shot isn't in the video any more")
        c = s["placed"][k]
        sc, f = self._shot_file(s, c)
        fps = float(f.get("fps") or 30.0) or 30.0
        fdur = float(f.get("dur") or (float(f["end"]) - float(f["start"])))
        a = max(0.0, float(c["src"]) - float(f["start"]))
        b = min(fdur, a + float(c["length"]))
        # the strip: the shot with room on both sides (3 s or 40% of it), at most 60 stills
        pad = max(3.0, 0.4 * (b - a))
        t0, t1 = max(0.0, a - pad), min(fdur, b + pad)
        step = max(1.0 / fps, (t1 - t0) / 60.0)
        stills = self._edit_stills(s["job"], f, t0, t1, step)
        room = self._shot_limits(s, k, c)
        return {"k": k, "job": s["job"], "what": str(c.get("what") or sc.get("what") or ""), "at": float(c["at"]), "length": float(c["length"]),
                "in": round(a, 3), "out": round(b, 3), "dur": round(fdur, 3), "fps": round(fps, 3), "max_length": round(room, 2),
                "min_length": self.MIN_SHOT_S, "t0": round(t0, 3), "t1": round(t1, 3), "step": round(step, 4), "stills": stills, "name": f.get("name")}

    def _edit_stills(self, job: str, f: dict, t0: float, t1: float, step: float) -> list[str]:
        """The stills of file f between t0 and t1 every `step` seconds, as names in the job folder (/media/<job>/<name>), made once."""
        try:
            stt = Path(f["path"]).stat()
        except OSError:
            return []
        key = hashlib.sha1(f"{f['path']}:{stt.st_size}:{int(stt.st_mtime)}:{t0:.3f}:{t1:.3f}:{step:.4f}".encode()).hexdigest()[:10]
        n = max(1, int((t1 - t0) / step) + 1)
        names = [f"se_{key}_{i:03d}.jpg" for i in range(n)]
        d = self.store.dir(job)
        if not all((d / x).exists() for x in names):
            try:
                subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{max(step, t1 - t0):.3f}", "-i", str(f["path"]),
                                "-an", "-vf", f"fps=1/{step:.4f}{self._lut_vf(f['path'])},scale=240:-2", "-q:v", "6", "-frames:v", str(n), "-start_number", "0", str(d / f"se_{key}_%03d.jpg")],
                               capture_output=True, timeout=240)
            except Exception as e:  # noqa: BLE001 - no stills is a quiet editor, not an error
                log.warning(f"no stills for the shot editor: {e}")
        return [x for x in names if (d / x).exists()]

    def _lut_vf(self, path) -> str:
        """The log-footage colour change as ",lut3d=..." for the end of a filter chain ("" for normal footage)."""
        try:
            from . import logcolor
            f = logcolor.vf(path, self.cfg)
            return ("," + f) if f else ""
        except Exception:  # noqa: BLE001
            return ""

    def shot_source_path(self, pid: str, k: int) -> Path:
        """The original file of placed shot k (the trim editor plays it as it is, the way Quick Look does)."""
        s = self._shot_state(pid)
        if not 0 <= k < len(s["placed"]):
            raise ValueError("that B-roll shot isn't in the video any more")
        return Path(self._shot_file(s, s["placed"][k])[1]["path"])

    def shot_preview(self, pid: str, k: int, a: float, b: float) -> Path | None:
        """A small playable copy (540p) of seconds a..b of shot k's source file, made once into the job folder - only for a file this page
        can't play as it is (the editor asks for it when the original fails to play)."""
        s = self._shot_state(pid)
        path = self.shot_source_path(pid, k)
        f = self._shot_file(s, s["placed"][k])[1]
        a = max(0.0, float(a))
        b = min(float(f.get("dur") or 1e9), max(a + 0.5, float(b)), a + 60.0)
        name = "sp_" + hashlib.sha1(f"{path}:{a:.2f}:{b:.2f}".encode()).hexdigest()[:12] + ".mp4"
        out = self.store.dir(s["job"]) / name
        if out.exists():
            return out
        tmp = out.with_suffix(".part.mp4")
        r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(path), "-t", f"{b - a:.3f}", "-an",
                            "-vf", "scale=-2:540" + self._lut_vf(path) + ",format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                            "-movflags", "+faststart", str(tmp)], capture_output=True, timeout=600)
        if r.returncode or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(out)
        return out

    def trim_shot(self, pid: str, k: int, src_in: float, src_out: float) -> dict:
        """The Edit button's Save: shot k shows `src_in`..`src_out` (seconds into its source file) from the same moment of the video. Nothing
        renders: the placement is changed and the video is marked out of date. A shot may be made longer only until the next one starts or
        the video ends; it never goes past either end of its file, and never under half a second. -> {in, out, length, clamped, pending}"""
        if self.running == pid:
            raise ValueError("this video is being made right now")
        s = self._shot_state(pid)
        if not 0 <= k < len(s["placed"]):
            raise ValueError("that B-roll shot isn't in the video any more")
        c = s["placed"][k]
        sc, f = self._shot_file(s, c)
        fps = float(f.get("fps") or 30.0) or 30.0
        fdur = float(f.get("dur") or (float(f["end"]) - float(f["start"])))
        try:
            a, b = float(src_in), float(src_out)
        except (TypeError, ValueError):
            raise ValueError("give the start and the end of the shot in seconds") from None
        if not (a == a and b == b):
            raise ValueError("give the start and the end of the shot in seconds")
        fr = lambda t: round(round(t * fps) / fps, 4)                      # noqa: E731 - whole frames
        clamped = min(a, b) < -1e-6 or max(a, b) > fdur + 1e-6               # (asked for film the file doesn't have)
        a, b = fr(max(0.0, min(a, fdur))), fr(max(0.0, min(b, fdur)))
        if b < a:
            a, b = b, a
        room = self._shot_limits(s, k, c)
        if b - a > room + 1e-6:                                              # the shot would run into the next one (or past the end)
            b, clamped = fr(a + room), True
        if b > fdur:
            b, clamped = fr(fdur), True
        if b - a < self.MIN_SHOT_S - 1e-6:                                    # never a flash: grow it to the minimum where the file has the film
            b, clamped = fr(min(fdur, a + self.MIN_SHOT_S)), True
            if b - a < self.MIN_SHOT_S - 1e-6:
                a, clamped = fr(max(0.0, b - self.MIN_SHOT_S)), True
        if b - a < self.MIN_SHOT_S - 1e-6:
            raise ValueError("that file is too short to show for half a second")
        self._stage_begin(s)
        placed = [dict(x) for x in s["placed"]]
        placed[k] = {**c, "src": round(float(f["start"]) + a, 3), "length": round(b - a, 3), "why": "trimmed by you"}
        rc = s["rc"]
        s["st"].put("render_cache.json", {**rc, "final": {**rc["final"], "cutaways": placed}, "pinned": True})
        n = self._stage_end(s["st"], "edited", "B-roll shot trimmed", float(c["at"]))
        return {"ok": True, "in": a, "out": b, "length": round(b - a, 3), "clamped": clamped, "pending": n}

    # -- "Don't use this" on a B-roll shot in the live view (avoid.py) ---------------------------------------------------------------
    def avoided(self, pid: str) -> list[dict]:
        return AV.load(State(home_of(self.projects.get(pid))))

    def avoid_shot(self, pid: str, shot: dict, undo: bool = False) -> dict:
        """Rule a shot out for this video -> {"result": "excluded" | "swapped" | "removed" | "next" | "noted" | "undone", "message"}.
        The shot is remembered, so the footage search, the B-roll picker and the Replace sheet never offer it again. What happens to it now
        depends on how far the video has come: before the B-roll is placed it is simply never picked ("excluded"); in a finished video it is
        swapped for the next best shot, or taken out when nothing fits, and the video renders again; while the render is already past
        placing B-roll nothing is changed under it - the shot goes out the next time it renders ("next")."""
        proj = self.projects.get(pid)
        st = State(home_of(proj))
        if undo:
            AV.forget(st, shot)
            return {"result": "undone", "message": "OK, it can be used again (the next time the B-roll is placed)."}
        avoid = AV.record(st, shot)
        if self.running == pid:
            job = self.store.get(proj["job"])
            if job.get("status") in ("transcribing", "finding") or (job.get("status") == "rendering" and float(job.get("progress") or 0) < 0.57):
                return {"result": "excluded", "message": "It won't be used in this video. The editor picks something else."}
            return {"result": "next", "message": "This video is already placing its B-roll, so it can't change under the render. "
                                                  "That shot will be left out the next time the video renders."}
        try:
            s = self._shot_state(pid)
        except ValueError:
            return {"result": "noted", "message": "It won't be used when this video is made."}
        bad = AV.bad_pool(avoid, s["pool"], s["files"])
        new, did = AV.swap(s["placed"], s["pool"], bad)
        if not (did["swapped"] or did["removed"]):
            return {"result": "noted", "message": "It isn't in the video, and it won't be offered again."}
        # Not rendered: the placement is changed (so the strip shows the new shot) and the video is marked out of date; Re-render is the person's.
        self._stage_begin(s)
        rc = s["rc"]
        s["st"].put("render_cache.json", {**rc, "final": {**rc["final"], "cutaways": new}})
        n = 0
        for c in s["placed"]:
            after = next((x for x in new if abs(float(x["at"]) - float(c["at"])) < 1e-6), None)
            if after is None:
                n = self._stage_end(s["st"], "removed", "B-roll shot ruled out and taken out", float(c["at"]))
            elif after.get("scene") != c.get("scene") or after.get("src") != c.get("src"):
                n = self._stage_end(s["st"], "avoided", "B-roll shot ruled out and swapped", float(c["at"]))
        n = n or self._stage_end(s["st"], "avoided", "B-roll shot ruled out", None)
        more = f" {n} change{'s' if n != 1 else ''} waiting: press Re-render when you're done."
        if did["removed"] and not did["swapped"]:
            return {"result": "removed", "message": "Taken out of the video (no other shot fitted there)." + more, "pending": n}
        return {"result": "swapped", "message": "Swapped for the next best shot." + more, "pending": n}

    def _review_state(self, pid: str):
        """(project, state, ai, sentences, plan) of a plan that is waiting for the creator's approval."""
        proj, folder, st, ai, engine = self._setup(pid)
        job = self.store.get(proj["job"])
        plan = st.get("plan.json")
        if job["status"] != "approval" or not plan or not plan.get("awaiting_approval"):
            raise ValueError("there is no plan waiting for review")
        _files, _group, _all, words = self.saved_inputs(folder, st, ai, job["options"].get("transcriber", "parakeet"))
        return proj, st, ai, seg(words), plan

    def review_edit(self, pid: str, op: dict) -> None:
        """One click on the review screen (drop / extend / add a line): the plan changes, its map and checks are rebuilt; nothing renders."""
        with self.review_lock:
            proj, st, ai, sents, plan = self._review_state(pid)
            outline = FL.steps(proj.get("notes") or "")
            if not outline:
                raise ValueError("this project has no script to review")
            if not plan.get("review_snapshot"):
                st.snapshot("plan.json", plan, "before-review-edits")
                plan["review_snapshot"] = True
            plan["items"] = FL.edit_items(sents, plan["items"], op, len(outline))
            plan["flow_edited"] = True
            plan["script_map"] = FL.script_map(sents, plan["items"], outline, None)
            plan["pending_render"], plan["pending_note"] = True, "edited in the review"
            st.put("plan.json", plan)

    def review_find(self, pid: str, step: int, query: str) -> list[dict]:
        proj, st, ai, sents, plan = self._review_state(pid)
        outline = FL.steps(proj.get("notes") or "")
        if not 1 <= step <= len(outline):
            raise ValueError("there is no such step")
        return FL.find_candidates(ai, sents, plan["items"], outline[step - 1], query)

    def approve(self, pid: str) -> None:
        """Render exactly the script map the creator just approved."""
        proj, folder, st, ai, engine = self._setup(pid)
        plan = st.get("plan.json")
        if not plan or not plan.get("awaiting_approval"):
            raise RuntimeError("there is no script map waiting for approval")
        st.snapshot("plan.json", plan, "approved-script-map")
        plan.pop("awaiting_approval", None)
        plan["approved_at"] = int(time.time())
        plan["pending_render"] = True
        plan["pending_note"] = "approved script map"
        st.put("plan.json", plan)
        self.rerender(pid)

    def revise_plan(self, pid: str, text: str) -> None:
        """Apply a creator's script-map note, then stop at the new map rather than rendering it unreviewed."""
        proj, folder, st, ai, engine = self._setup(pid)
        job = proj["job"]
        P = lambda status, stage, frac: self._prog(engine, job, status, stage, frac)       # noqa: E731
        plan = st.get("plan.json")
        if not plan or not plan.get("awaiting_approval"):
            raise RuntimeError("there is no script map waiting for revision")
        files, _group, _all_words, words = self.saved_inputs(folder, st, ai, self.store.get(job)["options"].get("transcriber", "parakeet"))
        sents = seg(words)
        P("finding", "Revising the script map", 0.4)
        ops = parse_request(ai, text, plan, self.cur_len(plan, sents, words))
        plan, note = self.apply(ops, plan, sents, words, ai, folder, st, text, P)
        outline = FL.steps(proj.get("notes") or "")
        if outline:
            plan["flow"] = FL.recheck(ai, sents, plan["items"], outline, old=plan.get("flow"))
            plan["script_map"] = FL.script_map(sents, plan["items"], outline, plan.get("flow"))
        plan["awaiting_approval"] = True
        plan.pop("flow_edited", None)
        plan["pending_render"], plan["pending_note"] = True, note
        st.snapshot("plan.json", st.get("plan.json"), "before-script-map-revision")
        st.put("plan.json", plan)
        self.store.update(job, status="approval", stage="Updated script map ready — review and approve", progress=0.54, error=None)

    def saved_inputs(self, folder: Path, st: State, ai, backend: str = "openai") -> tuple[list[dict], dict | None, list[dict], list[dict]]:
        """(files, multicam group, every word, the guest's words) exactly as the saved edit was planned from, so a re-render or a change
        means the same sentences. Edits made before plan_words.json existed are rebuilt the way they were made (never re-transcribed)."""
        files = st.get("files.json") or scan(folder)
        group = (st.get("multicam.json") or {}).get("group")
        pw = st.get("plan_words.json")
        if pw and pw.get("words"):
            from .parakeet import split_glued_numbers
            all_words = split_glued_numbers(pw["words"])      # (saved before the fix: "was22" -> "was" "22")
            return files, group, all_words, SP.drop_spans(all_words, [tuple(x) for x in pw.get("spans") or []])
        all_words = self.transcribe(files, ai, st, backend, lambda f, m: None, group, legacy=True)
        roles = st.get("roles.json")
        if roles and roles.get("spans") is not None:                       # the interviewer labels the edit was made with, as saved (an
            spans = [tuple(x) for x in roles["spans"]]                     # older version's cache key must not trigger a fresh labelling,
        else:                                                              # which could shift every sentence number of the saved edit)
            self.guest_words(all_words, ai, st)
            spans = [tuple(x) for x in (st.get("roles.json") or {}).get("spans", [])]
        st.put("plan_words.json", {"words": all_words, "spans": spans})   # from now on this edit's words are kept with it
        return files, group, all_words, SP.drop_spans(all_words, spans)

    def guest_words(self, words: list[dict], ai: LocalAI, st: State) -> list[dict]:
        """The transcript without the interviewer's questions (cached): only the person being interviewed can be picked."""
        sig = hashlib.sha1(f"v3:{len(words)}:{words[0]['start'] if words else 0}:{words[-1]['end'] if words else 0}".encode()).hexdigest()
        c = st.get("roles.json")
        if c and c.get("sig") == sig:
            spans = [tuple(x) for x in c["spans"]]
        else:
            spans = SP.host_spans(ai, words)
            st.put("roles.json", {"sig": sig, "spans": spans})
        kept = SP.drop_spans(words, spans)
        log.info(f"interviewer removed: {len(spans)} stretches, {len(words) - len(kept)} of {len(words)} words")
        return kept

    def find_group(self, files: list[dict], audios: list[dict], st: State, progress) -> dict | None:
        """Is this a multicam shoot? Cameras (and a recorder) matched by sound; cached until the files change."""
        sig = group_sig(files, audios)
        c = st.get("multicam.json")
        if c and c.get("sig") == sig:
            return c.get("group")
        try:
            group = MC.detect_group(files, audios, progress)
        except Exception as e:  # noqa: BLE001 - not multicam is a fine answer
            log.warning(f"multicam detection failed: {e}")
            group = None
        st.put("multicam.json", {"sig": sig, "group": group})
        return group

    @staticmethod
    def _song_music(song: str, old: dict | None = None) -> dict:
        """The plan's music for a song the person chose: exactly that file (kept through re-renders and changes)."""
        return {"mode": "auto", "file": str(song), "title": Path(song).stem, "song": True, "why": "your song", "mood": "", "genre": "",
                "used": sorted(set((old or {}).get("used", [])) | {Path(song).name})}

    def _music_choice(self, opts: dict, st: State) -> None:
        """The form's music setting -> the plan's: your song, auto, none, or a mood to aim for."""
        plan = st.get("plan.json") or {}
        m = opts.get("music")
        if opts.get("song"):
            plan["music"] = self._song_music(opts["song"], plan.get("music"))
            st.put("plan.json", plan)
            return
        if (plan.get("music") or {}).get("song"):                           # the song was taken off the project: choose again
            plan["music"] = {"mode": "auto"}
        if m == "":
            plan["music"] = {"mode": "none"}
        elif m not in (None, "auto"):
            plan["music"] = {"mode": "auto", "wanted": m}
        elif plan:
            plan["music"] = {k: v for k, v in (plan.get("music") or {"mode": "auto"}).items() if k != "mode" or v != "none"} or {"mode": "auto"}
            plan["music"]["mode"] = "auto"
        if plan:
            st.put("plan.json", plan)

    def _words_of(self, f: dict, ai: LocalAI, backend: str) -> list[dict]:
        """One file's words (file seconds), heard from its clean recording when one is synced to it."""
        from . import parakeet
        if backend != "parakeet":
            return self._openai_words(f, ai)
        if not parakeet.binary(self.cfg):                                   # (the default; a Mac without it uses OpenAI instead of failing)
            log.warning("Parakeet isn't installed on this Mac (clipper.ai_videos.parakeet_bin); transcribing with OpenAI")
            return self._openai_words(f, ai)
        clean = f.get("clean")
        if not clean or float(clean["offset"]) < 0:
            return parakeet.transcribe(Path(f["path"]), self.cfg)
        with tempfile.TemporaryDirectory(prefix="aivid-") as tmp:                     # the recorder's stretch that matches this video
            wav = Path(tmp) / "clean.wav"
            ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{float(clean['offset']):.3f}", "-t", f"{f['dur']:.3f}",
                        "-i", clean["audio"], "-vn", "-ac", "1", "-ar", "16000", str(wav)], timeout=1800, what="audio for transcription")
            return parakeet.transcribe(wav, self.cfg)

    def master_words(self, group: dict, ai: LocalAI, st: State, backend: str, progress, refresh: bool = False, legacy: bool = False) -> list[dict]:
        """Words on the multicam shoot's shared timeline, heard from the same sound the video uses: the recorder where there is one, else
        the camera whose own sound is cleanest (multicam.rate_sound) - each file transcribed once. legacy=True rebuilds the word list the
        way edits made before 2026-10-13 were planned (cameras in the order they start), from the cache only."""
        from .openai_audio import has_punctuation
        cache = st.get("words.json", {})

        def words_for(c: dict, n: int, total: int) -> list[dict]:
            p = Path(c["path"])
            key, sig = "mc:" + c["path"], [p.stat().st_size, int(p.stat().st_mtime)]
            ent = cache.get(key)
            if not (ent and ent.get("sig") == sig) or (refresh and not has_punctuation(ent["words"])):
                progress(n / max(1, total), f"Transcribing {p.name} ({n + 1} of {total})")
                ws = self._words_of({"path": c["path"], "dur": c["dur"], "clean": None}, ai, backend)
                ent = cache[key] = {"sig": sig, "words": ws}
                st.put("words.json", cache)
                if getattr(self, "live_job", None):
                    live.words(self.store, self.live_job, ws, of=p.name)
            return ent["words"]
        out: list[dict] = []
        if legacy:
            covered = -1e9
            clips = MC.sound_clips(group)
            for n, c in enumerate(clips):
                for w in words_for(c, n, len(clips)):
                    t = c["t0"] + w["start"]
                    if t >= covered - 0.05:
                        out.append({"text": w["text"], "start": round(t, 3), "end": round(c["t0"] + w["end"], 3)})
                covered = c["t0"] + c["dur"]
            return out
        spans = MC.sound_spans(group)
        paths = list(dict.fromkeys(c["path"] for c, _a, _b in spans))
        by_path = {}
        for n, path in enumerate(paths):
            c = next(c for c, _a, _b in spans if c["path"] == path)
            by_path[path] = words_for(c, n, len(paths))
        for c, x0, x1 in spans:
            for w in by_path[c["path"]]:
                s, e = c["t0"] + w["start"], c["t0"] + w["end"]
                if x0 - 1e-6 <= (s + e) / 2 < x1:
                    out.append({"text": w["text"], "start": round(s, 3), "end": round(e, 3)})
        return sorted(out, key=lambda w: w["start"])

    def transcribe(self, files: list[dict], ai: LocalAI, st: State, backend: str, progress, group: dict | None = None,
                   refresh: bool = False, legacy: bool = False) -> list[dict]:
        """Words for the whole virtual timeline. Cached per file (size + date), so a re-run only does new files. refresh=True (a full
        Make) also redoes cached transcripts that have no punctuation (made before it was put back) - a few cents per hour of talk."""
        from .openai_audio import has_punctuation
        cache = st.get("words.json", {})
        todo = [f for f in files if not f["broll"] and f["audio"] and not f.get("multicam")
                and not (cache.get(f["name"]) and cache[f["name"]].get("size") == f["size"] and cache[f["name"]].get("mtime") == f["mtime"]
                         and not (refresh and not has_punctuation(cache[f["name"]].get("words") or [])))]
        for done, f in enumerate(todo):
            progress(done / max(1, len(todo)), f"Transcribing {f['name']} ({done + 1} of {len(todo)})" + (" (Parakeet)" if backend == "parakeet" else ""))
            if getattr(self, "live_job", None) and f.get("path"):
                live.filmstrip(self.store, self.live_job, f["path"], f["dur"], of=f["name"], n=4)
            cache[f["name"]] = {"size": f["size"], "mtime": f["mtime"], "words": self._words_of(f, ai, backend)}
            st.put("words.json", cache)
            if getattr(self, "live_job", None):
                live.words(self.store, self.live_job, cache[f["name"]]["words"], of=f["name"])
        out: list[dict] = []
        for f in files:
            if f.get("multicam") and group:
                out += [{**w, "start": round(w["start"] + f["start"], 3), "end": round(w["end"] + f["start"], 3)}
                        for w in self.master_words(group, ai, st, backend, progress, refresh=refresh, legacy=legacy)]
                continue
            for w in (cache.get(f["name"]) or {}).get("words", []):
                out.append({"text": w["text"], "start": round(w["start"] + f["start"], 3), "end": round(w["end"] + f["start"], 3)})
        from . import glossary
        from .parakeet import split_glued_numbers
        return glossary.correct(split_glued_numbers(out), glossary.terms(self.cfg))   # names spelled the person's way

    def _openai_words(self, f: dict, ai: LocalAI) -> list[dict]:
        """One file's words from OpenAI, in overlapping chunks (a word on a chunk boundary is heard whole by one of them)."""
        from .openai_audio import chunk_windows, keep_share
        words: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="aivid-") as tmp:
            def one(k_win):
                k, (s, dur, keep_from, keep_to) = k_win
                mp3 = Path(tmp) / f"c{k}.mp3"
                clean = f.get("clean")
                src, shift = (clean["audio"], float(clean["offset"])) if clean and clean["offset"] + s >= 0 else (f["path"], 0.0)   # the clean recording, when synced
                ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{s + shift:.3f}", "-t", f"{dur:.3f}", "-i", src, "-vn",
                            "-ac", "1", "-ar", "16000", "-b:a", "48k", str(mp3)], timeout=1800, what="audio for transcription")
                return keep_share(ai.transcribe_mp3(mp3, dur), s, keep_from, keep_to)
            with ThreadPoolExecutor(max_workers=3) as ex:
                for part in ex.map(one, enumerate(chunk_windows(f["dur"], CHUNK_S))):
                    words += part
        return sorted(words, key=lambda w: w["start"])

    @staticmethod
    def sparse(f: dict, words: list[dict]) -> bool:
        """A file that is mostly pictures: few words per second, OR only a sliver of the time is spoken (a director's "go", "cut",
        "that was good" around a shot is not talking footage)."""
        if not f["audio"]:
            return True
        ws = [w for w in words if f["start"] <= w["start"] < f["end"]]
        if len(ws) / max(1.0, f["dur"]) < BR.SPARSE_WPS:
            return True
        spoken = sum(max(0.0, w["end"] - w["start"]) for w in ws)
        return spoken / max(1.0, f["dur"]) < 0.12

    def describe(self, files: list[dict], words: list[dict], ai: LocalAI, st: State, progress) -> dict[str, list[dict]]:
        """{file name: scenes in VIRTUAL seconds} for everything that is mostly pictures (B-roll, and any file with hardly any
        talking). A vision model looks at a few frames of each and says what it shows; looked at once, then cached."""
        from .glasses import contact_sheets
        cache = st.get("broll.json", {})
        look = [f for f in files if (f["broll"] or self.sparse(f, words)) and not f.get("library")]      # (library footage: described by the catalog)
        look = sorted(look, key=lambda f: -f["dur"])[:150] if len(look) > 150 else look                      # a sane cap on the AI bill
        todo = [f for f in look if not (cache.get(f["name"]) and cache[f["name"]].get("size") == f["size"]
                                        and cache[f["name"]].get("sheets", BR.MAX_SHEETS) >= BR.sheets_for(f["dur"]))]       # a long file is looked at more closely
        for n, f in enumerate(todo):
            progress(n / max(1, len(todo)), f"Looking at {f['name']} ({n + 1} of {len(todo)})")
            try:
                n_sheets = BR.sheets_for(f["dur"])
                step = max(1.0, f["dur"] / (12 * n_sheets))
                sheets = contact_sheets(Path(f["path"]), 0.0, f["dur"], step)[:n_sheets]
                if not sheets:
                    continue
                for b, ts in sheets[:2]:
                    self._live("sheet", f=live.frame(self.store, self.live_job, b, width=480) if getattr(self, "live_job", None) else "",
                               of=f["name"], frames=len(ts))
                times = ", ".join(f"{t:.1f}" for _, ts in sheets for t in ts)
                prompt = (f"File: {f['name']} ({f['dur']:.1f} s long). {len(sheets)} contact sheet(s); frame times in seconds: {times}. "
                          "Describe the scenes. Use these times.")
                res = ai.complete_json("broll_describe", BR.SYSTEM, prompt, BR.SCHEMA, images=[b for b, _ in sheets])
                sc = BR._valid_scenes(res.data.get("scenes"), 0.0, f["dur"])
            except Exception as e:  # noqa: BLE001
                log.warning(f"describing {f['name']} failed: {e}")
                continue
            if sc:
                cache[f["name"]] = {"size": f["size"], "scenes": sc, "sheets": n_sheets}
                st.put("broll.json", cache)
                for x in sc[:12]:
                    self._live("scene", of=f["name"], start=round(float(x.get("start") or 0), 1), end=round(float(x.get("end") or 0), 1),
                               what=str(x.get("what") or "")[:200], quality=x.get("quality"))
        out = {}
        looked = {f["name"] for f in look}
        for f in files:
            c = cache.get(f["name"])
            if f["name"] in looked and c:
                out[f["name"]] = BR.split_long([{**s, "start": round(s["start"] + f["start"], 2), "end": round(s["end"] + f["start"], 2)} for s in c["scenes"]])
        lib = {**((st.get("library.json") or {}).get("scenes") or {}), **((st.get("stock.json") or {}).get("scenes") or {})}
        names = {f["name"] for f in files if f.get("library")}
        out.update({k: v for k, v in lib.items() if k in names})
        return out

    def plan(self, files, words, sents, scenes, ai, st, opts, brief, pid, progress) -> dict:
        """Decide what to keep: a list of items (talk = a run of sentences, visual = a stretch of mostly-silent footage)."""
        notes = (self.projects.get(pid).get("notes") or "")                  # WHAT THE CREATOR WROTE (the brief also carries the editor's learned rules)
        style = video_kind(ai, notes, opts.get("kind"))                      # an ad is cut very differently from a story
        want_pace = opts.get("pace") or "natural"
        if want_pace == "natural" and (style in ("promo", "ad") or PROFILES.get(style, {}).get("pace") == "tight"):   # Automatic: an energetic kind cuts pauses tighter
            want_pace = "tight"
        gap = float(PACES.get(want_pace, PACES["natural"])["gap"])
        named = {k for k, f in enumerate(files) if f["broll"] and scenes.get(f["name"])}                  # B-roll: cutaway footage only
        silent = {k for k, f in enumerate(files) if not f["broll"] and scenes.get(f["name"]) and self.sparse(f, words)}   # mostly pictures
        pool = named | silent
        if len(pool) >= len(files):
            pool, named, silent = set(), set(), set()
        sfiles = [dict(f, broll=(k in pool)) for k, f in enumerate(files)]
        talk_s = sum(f["dur"] for k, f in enumerate(files) if k not in pool and f["audio"] and not f["broll"])
        target = float(opts.get("long_minutes") or 0) * 60                   # the length box wins; else what the notes ask for
        if not target:
            target = length_from_notes(ai, notes)
        total = max(f["end"] for f in files if not f.get("library"))
        if not target and talk_s > 900:
            target = max(60.0, min(600.0, talk_s * 0.1))                  # a lot of footage and no length given: ~10%, at most 10 min
        eng = type("E", (), {"creative_router": lambda self_: ai})()
        old = st.get("plan.json", {}) or {}
        outline = FL.steps(notes)                                          # the creator's script: a REQUIREMENT, followed step by step
        flow_report = None
        if outline:
            progress(0.0, f"Following your outline ({len(outline)} steps)")
            budget = target or max(150.0, min(300.0, 0.1 * talk_s))
            items, title, summary, flow_report = FL.plan(ai, sents, outline, notes, budget, old.get("title", ""))
            if not items:
                raise RuntimeError("the AI could not find anything for your outline in the transcript")
            silent_scenes = []
            progress(0.7, "Checking every step of the outline")
        elif target and target < 0.25 * talk_s:                             # a short video from a lot of footage: best moments
            progress(0.0, "Finding the best moments")
            cands = HL.candidates(ai, sents, sfiles, brief, style=style,
                                  progress=lambda i, n: progress(0.6 * i / max(1, n), f"Reading the footage ({i} of {n})"))
            progress(0.7, "Choosing and ordering the moments")
            cands = self.mark_sound(cands, sfiles, st)                     # moments recorded badly: marked, scored down, never the opening
            ranges, title, summary = HL.assemble(ai, cands, max(15.0, target - OUTRO_S - 3.0) if style == "ad" else target,
                                                 brief + (SOUND_NOTE if any(c.get("sound") for c in cands) else ""), old.get("title", ""), style=style)
            if not ranges:
                raise RuntimeError("the AI found no moments worth keeping - add notes about what you want, or a longer target length")
            items = [{"kind": "talk", "a": a, "b": b} for a, b in ranges]
            items = self.top_up(ai, sents, sfiles, items, target, gap, words, brief, style, progress)
            silent_scenes = []                                              # a short cut: silent footage only as cutaways
        else:
            progress(0.0, "Editing the story (keeping it in order)")
            tgt = target or total * 0.85
            sp = story.plan(eng, words, sfiles, total, old.get("title") or "Video", tgt, gap, brief)
            title, summary = sp.title, sp.summary
            items = items_from_intervals(sp.intervals, sents)
            silent_scenes = None
        # the silent footage the AI has now SEEN: pick the shots worth playing on their own; the rest become cutaways
        vis = [{**s, "file": files[k]["name"]} for k in sorted(silent) for s in scenes.get(files[k]["name"], []) if s["quality"] >= 3]
        picked: list[int] = []
        if vis and silent_scenes is None:
            talk_len = sum(b - a for a, b in self.intervals_of({"items": items, "gap": gap}, sents, words))
            picked = BR.pick_standalone(ai, vis, brief, summary, max(10.0, 0.5 * talk_len))
            talk_items = [it for it in items if it["kind"] == "talk"]
            shots = [{"kind": "visual", "start": vis[i]["start"], "end": min(vis[i]["end"], vis[i]["start"] + BR.STANDALONE_MAX_S)} for i in picked]
            items = merge_items(talk_items, [it for it in items if it["kind"] == "visual"] + shots, None, {s.idx: s for s in sents})
        pool_scenes = [s for i, s in enumerate(vis) if i not in set(picked)] + \
            [{**s, "file": files[k]["name"]} for k in sorted(named) for s in scenes.get(files[k]["name"], [])]
        try:                                                                   # photos in the B-roll folders: the AI looks at each once
            folder = Path(self.projects.get(pid)["folder"])
            cache = st.get("photos.json", {}) or {}
            found = PH.describe(ai, PH.scan(folder, skipped), cache)
            st.put("photos.json", cache)
            pool_scenes = pool_scenes + PH.scenes(found)
            if found:
                log.info(f"{len(found)} photo(s) offered as B-roll")
        except Exception as e:  # noqa: BLE001 - photos are a bonus, never a reason to fail
            log.warning(f"looking at the photos failed: {e}")
        if any(f.get("library") and not f.get("stock") for f in files):
            pool_scenes = pool_scenes + ((st.get("library.json") or {}).get("photos") or [])
        if opts.get("stock"):                                                  # stock photos (only where a search found no video)
            pool_scenes = pool_scenes + ((st.get("stock.json") or {}).get("photos") or [])
        plan = {"style": style, "items": items, "pool": pool_scenes, "title": title or old.get("title") or "My video", "summary": summary, "gap": gap,
                "hook_text": old.get("hook_text", ""), "music": old.get("music") or {"mode": "auto"}, "beat_sync": bool(old.get("beat_sync")),
                "target_s": target or None, "version": int(old.get("version", 0)), "history": old.get("history", [])}
        if flow_report:
            plan["flow"] = flow_report
        chosen = {i for it in plan["items"] if it["kind"] == "talk" for i in range(it["a"], it["b"] + 1)}
        bad = SP.check_selected(ai, sents, chosen)                          # a last look: no interviewer questions inside what was kept
        if bad:
            log.info(f"{len(bad)} interviewer sentence(s) taken out of the edit")
            plan["items"] = drop_sentences(plan["items"], bad)
        plan["items"] = no_repeats(plan["items"], sents)                    # a line is said once
        if flow_report:                                                      # the script supervisor: check the cut against every point of the script, repair it
            progress(0.74, "Checking the cut against your script")
            banned = set(bad)

            def clean(its):                                                  # the interviewer, kept out of everything the supervisor adds or extends
                ch = {i for it in its if it["kind"] == "talk" for i in range(it["a"], it["b"] + 1)}
                more = SP.check_selected(ai, sents, ch)
                if more:
                    log.info(f"{len(more)} more interviewer sentence(s) taken out after the script check")
                    banned.update(more)
                    its = no_repeats(drop_sentences(its, more), sents)
                return its
            plan["items"], plan["flow"] = FL.supervise(ai, sents, plan["items"], outline, notes, budget, clean=clean, banned=banned)
            plan["items"] = finish_sentence(plan["items"], sents)
        if style in ("promo", "ad"):
            by_idx = {s.idx: s for s in sents}
            sure = [it for it in plan["items"] if it["kind"] != "talk" or not any(
                HEDGE_RE.search(by_idx[i].text) for i in range(it["a"], it["b"] + 1) if i in by_idx)]
            if any(it["kind"] == "talk" for it in sure):
                plan["items"] = sure                                         # an ad only says confident things
        plan["items"] = finish_sentence(plan["items"], sents)                # never end on half a sentence
        if style == "ad" and target:                                         # the ad is as long as asked: the end card comes on top of the talking
            room = max(15.0, target - OUTRO_S) * 1.06
            while True:
                talk = [it for it in plan["items"] if it["kind"] == "talk"]
                lens = [(sum(b - a for a, b in self.intervals_of({"items": [it], "gap": gap}, sents, words)), it) for it in talk]
                if sum(x for x, _ in lens) <= room or len(talk) <= 3:
                    break
                drop = max(lens[1:-1] or lens, key=lambda x: x[0])[1]       # the longest one in the middle (the opening and the close stay)
                plan["items"] = [it for it in plan["items"] if it is not drop]
        if style in ("promo", "ad"):
            if (plan["music"] or {}).get("mode") != "none" and not plan["music"].get("wanted") and not plan["music"].get("file"):
                plan["music"] = {**plan["music"], "style_wanted": "upbeat"}          # only if you have no songs of your own in Music/
            if "beat_sync" not in old:
                plan["beat_sync"] = True                                    # fast cuts that land on the music
        if not plan["hook_text"]:
            plan["hook_text"] = self.hook(ai, plan, sents, brief)
        plan["pending_render"], plan["pending_note"] = True, "first cut" if not plan.get("version") else "made again"
        if outline:
            # Stop here. The creator can see every selected sentence before any media is rendered.
            plan["script_map"] = FL.script_map(sents, plan["items"], outline, plan.get("flow"))
            plan["awaiting_approval"] = True
        old_plan = st.get("plan.json")
        if old_plan:
            st.snapshot("plan.json", old_plan, "before-new-script-map")
        st.put("plan.json", plan)                                           # saved BEFORE rendering: a failed render is redone from this edit
        return plan

    def top_up(self, ai, sents, sfiles, items: list[dict], target: float, gap: float, words, brief: str, style: str, progress) -> list[dict]:
        """The creator asked for about `target` seconds; if the picked moments come to much less (the AI was given only a handful of candidates, or
        some were dropped as repeats / questions), look again with a wider net and add the best unused moments - each placed in time order among
        the others - until the edit is at least ~90% of the asked length. Never adds anything that overlaps what is already kept."""
        if not target or style == "ad":
            return items

        def spoken(its):
            return sum(b - a for a, b in self.intervals_of({"items": its, "gap": gap}, sents, words))
        have = spoken(items)
        if have >= 0.9 * target:
            return items
        progress(0.62, f"The cut is {have:.0f} s of the {target:.0f} s asked for - finding more moments")
        by = {s.idx: s for s in sents}
        used = {i for it in items for i in range(it["a"], it["b"] + 1)}
        cands = HL.candidates(ai, sents, sfiles, brief, per_chunk=10, style=style)
        added = 0
        for c in sorted(cands, key=lambda c: -c["score"]):
            idxs = set(range(c["first"], c["last"] + 1))
            if idxs & used or have >= 0.97 * target:
                continue
            ln = sum(by[i].end - by[i].start for i in idxs if i in by) + gap
            if have + ln > target * 1.1:
                continue
            new = {"kind": "talk", "a": c["first"], "b": c["last"]}
            at = next((k for k, it in enumerate(items) if it["a"] > new["b"]), len(items))
            if at < len(items) and any(it["a"] < new["a"] for it in items[at:]):       # the order of the edit isn't chronological: add at the end
                at = len(items)
            items.insert(at, new)
            used |= idxs
            have += ln
            added += 1
        log.info(f"top-up: added {added} moment(s); the cut is now about {have:.0f} s (asked for {target:.0f} s)")
        return items

    def hook(self, ai, plan: dict, sents, brief: str, note: str = "") -> str:
        by = {s.idx: s for s in sents}
        opening = " ".join(by[it["a"]].text for it in plan["items"][:2] if it["kind"] == "talk" and it["a"] in by)[:600]
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        prompt = (f"Video: {plan['title']}\nWhat it says: {plan['summary']}\nHow it opens: {opening}\n"
                  + (f"Creator's notes: {brief[:800]}\n" if brief else "") + (f"Change requested: {note}\n" if note else "")
                  + "Write the on-screen hook: 3 to 9 words, a promise or question that makes a viewer stay, no hashtags, no quotes, no emoji.")
        try:
            res = ai.complete_json("hook_text", "You write short, punchy on-screen hooks for videos.", prompt, schema)
            return " ".join(str(res.data.get("text", "")).split()).strip("\"' ")[:70]
        except Exception as e:  # noqa: BLE001
            log.warning(f"hook text failed: {e}")
            return ""

    # -- music + beats -----------------------------------------------------------------------
    def energetic(self, st: State, exclude: set[str], seed: int):
        """For an ad: a driving, upbeat/epic track around 110-140 BPM (tempo measured, not guessed from the name)."""
        cands = []
        for t in MU.library(self.cfg):
            if not ({"upbeat", "epic"} & set(t.moods)) or t.path.name in exclude:
                continue
            g = self.beat_grid(t, st)
            bpm = (g or {}).get("bpm") or 0
            if 100 <= bpm <= 150:
                cands.append((abs(bpm - 126), t.path.name, t))
        cands.sort(key=lambda c: (c[0], c[1]))
        return cands[seed % min(4, len(cands))][2] if cands else None

    def choose_track(self, plan: dict, folder: Path, ai, words: list[dict], title: str, force_new: bool = False, want: str = "",
                     st: State | None = None):
        m = plan.get("music") or {"mode": "auto"}
        if m.get("mode") == "none":
            return None
        if m.get("song") and m.get("file") and not force_new:              # your song: exactly that one
            t = MU.song(m["file"])
            if t:
                return t
            log.warning(f"the chosen song isn't there any more: {m['file']}")
        if m.get("file") and not force_new:
            t = MU.resolve(self.cfg, m["file"])
            if t:
                return t
        used = set(m.get("used", []))
        exclude = used if force_new else set()
        want = want or str(m.get("wanted") or "")                            # what YOU asked for (the form or a change request)
        own = [t for t in MU.own_tracks(folder / "Music") if t.path.name not in exclude] or MU.own_tracks(folder / "Music")
        if own and not want:
            seed = int(hashlib.sha1((title + str(len(used))).encode()).hexdigest()[:6], 16)
            return own[seed % len(own)]
        if plan.get("style") == "ad" and not want and st is not None:
            t = self.energetic(st, exclude, int(hashlib.sha1((title + str(len(used))).encode()).hexdigest()[:6], 16))
            if t:
                plan["music"] = {**m, "mood": "upbeat", "why": "a driving track for an ad"}
                return t
        want = want or str(m.get("style_wanted") or "")                      # no songs of yours: aim for what suits the kind of video
        text = " ".join(w["text"] for w in words)[:6000]
        genres = MU.genres(self.cfg)
        if want and want.lower() in MU.MOODS:
            vibe = {"mood": want.lower(), "genre": "", "why": f"you asked for {want}", "by": "you"}
        else:
            vibe = MU.choose_vibe(ai, text + (f"\nThe creator wants the music to be: {want}" if want else ""), genres, title)
        t = MU.pick(self.cfg, mood=vibe["mood"], genre=vibe.get("genre", ""), seed=f"{title}-{len(used)}", exclude=exclude)
        plan["music"] = {**m, "mood": vibe["mood"], "genre": vibe.get("genre", ""), "why": vibe.get("why", "")}
        return t

    def beat_grid(self, track, st: State) -> dict | None:
        cache = st.get("beats.json", {})
        key = f"{track.path.name}:{track.path.stat().st_size}"
        if key not in cache:
            cache[key] = BT.analyze(track.path) or {}
            st.put("beats.json", cache)
        return cache[key] or None

    # -- render ------------------------------------------------------------------------------
    def intervals_of(self, plan: dict, sents: list[M.Sentence], words: list[dict], all_words: list[dict] | None = None,
                     with_bounds: bool = False):
        """The kept stretches (virtual seconds) of the plan's items, in play order. Each stretch is held inside the gap around its own
        words: it may breathe into the silence, but never reaches into the word said just before or just after it (checked against EVERY
        word, the interviewer's too). with_bounds=True also returns those limits [(earliest start, latest end)], for tightening."""
        by = {s.idx: s for s in sents}
        ref = sorted(all_words or words, key=lambda w: w["start"])
        starts = [w["start"] for w in ref]
        out: list[tuple[float, float]] = []
        bounds: list[tuple[float, float]] = []
        hand_cuts = TE.cut_spans(plan, ref) if plan.get("te_cuts") else []
        for it in plan["items"]:
            if it["kind"] == "visual":
                out.append((it["start"], it["end"]))
                bounds.append((-1e9, 1e9))
            elif it["a"] in by and it["b"] in by:
                s0, s1 = float(it.get("t0", by[it["a"]].start)), float(it.get("t1", by[it["b"]].end))    # (t0/t1: words added by hand, transcript_edit.py)
                for a, b in repace(words, s0, s1, float(plan.get("gap", 0.75))):
                    lo, hi = word_limits(ref, starts, words, max(a, s0), min(b, s1))    # its own words: inside its sentences only
                    a2, b2 = max(a, lo), min(b, hi)
                    for a3, b3 in (TE.subtract(a2, b2, hand_cuts) if hand_cuts else [(a2, b2)]):    # (words taken out in the transcript panel)
                        if b3 - a3 > 0.05:
                            out.append((a3, b3))
                            bounds.append(word_limits(ref, starts, words, a3, b3) if hand_cuts and (a3, b3) != (a2, b2) else (lo, hi))   # (a piece beside a hand cut: held clear of the cut words)
        return (out, bounds) if with_bounds else out

    def render(self, pid: str, plan: dict, files, words, scenes, ai: LocalAI, engine: LocalEngine, st: State, folder: Path, note: str,
               all_words: list[dict] | None = None) -> None:
        proj = self.projects.get(pid)
        job = proj["job"]
        NT.begin()                                                           # (what this render finds goes into Notes/Quality check.md)
        P = lambda status, stage, frac: self._prog(engine, job, status, stage, frac)       # noqa: E731
        opts = self.store.get(job)["options"]
        if all_words is None:                                               # every word, the interviewer's too: what a cut must never reach into
            all_words = (st.get("plan_words.json") or {}).get("words") or words
        sents = seg(words)
        self.rate_sound(st)                                                  # which camera's own recording is the cleanest (used for ALL the sound)
        mcj = st.get("multicam.json") or {}
        refresh_timecodes(files, mcj.get("group"))
        if mcj.get("group"):
            st.put("multicam.json", mcj)                                     # the corrected timecodes are what the build below reads back
        keep = plan.get("keep_cut")
        mont = plan.get("montage")
        if mont:                                                               # a montage / slideshow: the photos become clips, the shots play in the plan's order
            files = [f for f in files if not f.get("mphoto")] + self.bake_montage(plan, st)
            intervals = [(float(i["start"]), float(i["end"])) for i in plan["items"] if i.get("kind") == "visual"]
        elif keep:                                                             # "Add B-roll to a video": the whole video, once, as delivered
            intervals = [(float(keep["start"]), float(keep["end"]))]
        else:
            raw, bounds = self.intervals_of(plan, sents, words, all_words, with_bounds=True)
            intervals = self.tighten(raw, files, st, bounds)
            if self.cfg.get("clipper.ai_videos.cut_fillers", True):
                intervals = cut_fillers(intervals, words)                      # "um", "uh" and stuttered repeats ("I I") are taken out of the speech
            intervals = dedupe_intervals(intervals)                            # no moment is played twice
            intervals = merge_tiny(intervals, all_words)                       # a sliver of talk (under 1.2 s) joins its neighbour: no flash
            from . import negation
            intervals, kept_not = negation.keep_negations(intervals, words)    # no cut flips a sentence ("did not like" -> "did like")
            for n_ in kept_not:
                log.info(f"meaning: {n_}")
            for s_ in negation.lost_negations(intervals, words):
                log.warning(f"meaning check: a kept sentence lost its negation: {s_[:160]}")
            grades = [self.sound_of(files, (st.get("multicam.json") or {}).get("group"), st, a, b) for a, b in intervals]
            intervals, gone = opening_without_bad_sound(intervals, grades)     # never open on a badly recorded line
            if gone:
                log.info(f"sound: {len(gone)} badly recorded stretch(es) taken out of the opening")
                NT.note("info", f"{len(gone)} badly recorded line{'s' if len(gone) != 1 else ''} kept out of the opening (far from the mic, or muffled)",
                        do="Nothing: the video opens on cleanly recorded lines.")
            self.save_grades(st)
        if not intervals:
            raise RuntimeError("nothing was kept - the edit came out empty")
        picked = opts.get("gfx_style")
        if picked and picked != "auto":                                  # the look the person picked wins (also over an earlier make's)
            from . import graphics as GR
            if picked in GR.THEMES:
                plan["theme"], plan["theme_why"] = picked, "picked"
        elif plan.get("theme_why") == "picked":                          # ...and "matches the footage" again chooses again
            plan["theme"] = None
        if not plan.get("theme"):
            from . import graphics as GR
            talk = next((f for f in files if not f["broll"] and not f.get("multicam") and f.get("path")), None) or next((f for f in files if f.get("path")), None)
            gf = next((c for a in ((st.get("multicam.json") or {}).get("group") or {}).get("angles", []) for c in a["files"]), None)      # a multicam shoot: its first camera
            src_luma = talk or gf
            luma = GR.video_luma(src_luma["path"], float(src_luma["dur"])) if src_luma else None             # how bright the footage is
            plan["theme"] = GR.match_theme(luma, plan.get("style") or "story", pid)           # one look for the whole video, matched to its footage and kind
            plan["theme_why"] = f"footage brightness {luma if luma is not None else '?'}, {plan.get('style') or 'story'}"
        # music, and (if asked) cutting to its beat
        P("rendering", "Choosing the music", 0.55)
        track = self.choose_track(plan, folder, ai, words, plan["title"], st=st)
        music_start, beat_note = 0.0, ""
        if track and getattr(self, "live_job", None):
            why = (plan.get("music") or {}).get("why") or (plan.get("music") or {}).get("mood") or ""
            live.stage(self.store, self.live_job, f"Music: {track.title}" + (f" ({why})" if why else ""))
        if track:
            plan["music"] = {**(plan.get("music") or {}), "file": str(track.path), "title": track.title,
                             "used": sorted(set((plan.get("music") or {}).get("used", [])) | {track.path.name})}
            grid = self.beat_grid(track, st) if plan.get("beat_sync") else None
            if grid and mont:                                              # the shots were already laid out in whole beats / bars: the song starts on its first beat
                music_start = max(0.0, grid["first"])
                beat_note = f"{grid['bpm']:.0f} BPM, every cut on the beat"
            elif grid:
                music_start = max(0.0, grid["first"])                      # the song begins ON a beat, so beats fall on k * period
                out_len = sum(b - a for a, b in intervals)
                bt = BT.grid(grid["period"], 0.0, out_len + 2)
                intervals, n = BT.snap_intervals(intervals, all_words, bt)      # every word: a cut never moves onto anyone's speech
                beat_note = f"{grid['bpm']:.0f} BPM, {n} cuts moved onto the beat"
            elif plan.get("beat_sync"):
                beat_note = "that song has no steady beat to cut to"
        out_len = sum(b - a for a, b in intervals)
        st.put("plan.json", {**plan, "pending_render": True, "pending_note": plan.get("pending_note") or note})   # a re-render keeps this song
        # B-roll cutaways over the talking
        bbrief = self.projects.brief(proj) + ("\nWhat the editor asked for in the B-roll: " + "; ".join(plan["broll_notes"][-6:])
                                              if keep and plan.get("broll_notes") else "")
        pool = plan.get("pool")
        if mont:
            pool = []                                                       # (every shot is in the plan already: nothing is laid over anything)
        if pool is None:                                                    # an older plan: the B-roll files only
            pool = [{**s, "file": f["name"]} for f in files if f["broll"] for s in scenes.get(f["name"], [])]
        ad = plan.get("style") == "ad"
        pool = [s for s in pool if s["quality"] >= (6 if ad else 4)]
        cutaways: list[dict] = []
        # what was said while no camera was rolling (only a recorder heard it): covered by B-roll from start to end, or not used at all
        group = (st.get("multicam.json") or {}).get("group")
        nopic = no_picture_spans(files, group, intervals) if group and not keep else []
        if nopic and len(pool) < sum(max(1, math.ceil((b - a + 0.3) / 2.0)) for a, b in nopic):
            intervals = subtract_spans(intervals, nopic)
            log.info(f"{len(nopic)} stretch(es) that no camera filmed left out (not enough B-roll to cover them)")
            NT.note("info", f"{len(nopic)} stretch{'es' if len(nopic) != 1 else ''} only the sound recorder heard (no camera was rolling) left out: "
                    "not enough B-roll to cover them", do="Add more B-roll to the folder if those lines matter, and make it again.")
            nopic = []
            if not intervals:
                raise RuntimeError("everything kept was said while no camera was filming, and there is no B-roll to show over it")
            out_len = sum(b - a for a, b in intervals)
        pace = str(opts.get("pace") or "natural")
        per_s = float(plan.get("broll_per_s") or PROFILES.get(plan.get("style"), PROFILES["story"])["cut_per_s"]) * PACE_BROLL.get(pace, 1.0)
        # The same edit rendered again gets the SAME B-roll (the AI that places it is not repeatable, and a re-render used to lose or move shots you
        # had already looked at). The placement is remembered for this exact cut + B-roll pool; a changed cut / pool / a new plan places it afresh.
        iv_fp = [[round(a, 2), round(b, 2)] for a, b in intervals]
        pool_fp = [[x.get("file"), round(x["start"], 1), round(x["end"], 1), x.get("what", "")[:40]] for x in pool]
        rsig = hashlib.sha1(json.dumps(iv_fp + pool_fp).encode()).hexdigest()
        rc = st.get("render_cache.json") or {}
        # A placement swapped by hand (pinned) also stays when the pool only GREW since it was saved (a shot from the footage library added
        # in the Replace sheet): the saved signature is checked against the same cut and the first pool_n shots of today's pool.
        n0 = rc.get("pool_n")
        grown = bool(rc.get("pinned")) and isinstance(n0, int) and 0 < n0 <= len(pool) and \
            hashlib.sha1(json.dumps(iv_fp + pool_fp[:n0]).encode()).hexdigest() == rc.get("sig")
        # An ad's B-roll is placed afresh on each render (its rhythm is the point), unless someone swapped a shot by hand: then it stays.
        reuse = (not ad or bool(rc.get("pinned"))) and (rc.get("sig") == rsig or grown) and rc.get("planned") is not None
        carried = False
        if reuse:
            cutaways = [dict(c) for c in rc["planned"]]
        elif pool and not ad and rc.get("te_carry"):                        # a transcript edit: the shots whose stretch is still there stay (transcript_edit.py)
            cutaways, carried = TE.carry_cutaways(rc["te_carry"], intervals), True
        if pool and not reuse and not carried:
            P("rendering", "Placing B-roll cutaways", 0.58)
            cutaways = BR.plan_cutaways(ai, seg(remap_words(words, intervals, 1.0)), pool,
                                        bbrief, out_len, ad=ad, per_s=per_s)                  # (fast-paced: more often; take it slow: less)
            allowed = BR.ad_suitable(ai, pool, self.projects.brief(proj)) if ad else None
            if ad:
                cutaways = BR.fill_ad(cutaways, seg(remap_words(words, intervals, 1.0)), pool, out_len, allowed=allowed)
            ph = [i for i, s in enumerate(pool) if s.get("photo") and i not in {c["scene"] for c in cutaways}]
            if ph:                                                          # the photos, each where it fits what is being said
                extra = BR.plan_cutaways(ai, seg(remap_words(words, intervals, 1.0)), [pool[i] for i in ph], bbrief, out_len, ad=ad,
                                         per_s=12.0 * PACE_BROLL.get(pace, 1.0))
                for c in extra:
                    if all(abs(c["at"] - o["at"]) > o["length"] + 2.0 for o in cutaways):
                        cutaways.append({**c, "scene": ph[c["scene"]]})
                cutaways.sort(key=lambda c: c["at"])
            if plan.get("beat_sync") and track and beat_note and "BPM" in beat_note:
                grid = self.beat_grid(track, st)
                bt = BT.grid(grid["period"], 0.0, out_len + 2)
                for c in cutaways:
                    c["at"] = round(min(bt, key=lambda t: abs(t - c["at"])), 2)
            if not ad:
                cutaways = thin_cutaways(cutaways, pace)                    # room to breathe between shots (unless fast-paced)
            if nopic:
                cutaways, missed = cover_spans(cutaways, [(out_time(intervals, a), out_time(intervals, b)) for a, b in nopic], pool)
                if missed:
                    log.warning(f"{len(missed)} stretch(es) no camera filmed could not be covered")
                    for a, b in missed[:6]:
                        NT.note("warn", "a stretch no camera filmed (only the recorder heard it) isn't fully covered by B-roll",
                                start=a, end=b, do="Put a B-roll shot or a photo over it in Final Cut.")
        # an ad ends on a B-roll shot with the logo, the name and a call to action (the voice has finished by then)
        outro = None
        if ad and pool:
            spare = [(i, s) for i, s in enumerate(pool) if i not in {c["scene"] for c in cutaways} and s["end"] - s["start"] >= OUTRO_S and not s.get("photo")]
            if spare:
                i, sc = max(spare, key=lambda x: x[1]["quality"])
                bf = next((f for f in files if not f.get("multicam") and f["start"] - 1e-3 <= sc["start"] < f["end"]), None)
                if bf:
                    outro = {"scene": i, "path": bf["path"], "ss": sc["start"] - bf["start"], "d": OUTRO_S, "w": bf["w"], "h": bf["h"], "fps": bf["fps"],
                             "tc": bf.get("tc", 0.0), "file_dur": bf["dur"]}
        # multicam: the director picks the camera angle for every kept stretch of the shoot
        cuts: list[dict] = []
        if group:
            P("rendering", "Directing the camera angles", 0.59)
            cuts = self.direct(group, intervals, ai, words, self.projects.brief(proj), st)
        # the twelve-second rule: something changes on screen at least every 12 seconds (another camera, B-roll or a photo)
        sents_out = seg(remap_words(words, intervals, 1.0))
        # Where the script moves to the next creator-defined step, make the otherwise inevitable interview jump feel intentional.
        by_sentence = {s.idx: s for s in sents}
        step_boundaries, prior_step = [], None
        for it in plan["items"]:
            if it.get("kind") != "talk" or not it.get("step"):
                continue
            if prior_step is not None and it["step"] != prior_step and it["a"] in by_sentence:
                step_boundaries.append(by_sentence[it["a"]].start)
            prior_step = it["step"]

        def choose(t: float, free: list[int]):
            spoken = " ".join(s.text for s in sents_out if s.end >= t - 2.0 and s.start <= t + 6.0)
            return BR.pick_for(ai, spoken, pool, free, bbrief)
        planned = [dict(c) for c in cutaways]
        csig = hashlib.sha1(json.dumps([[round(c["start"], 2), round(c["end"], 2), c["angle"]] for c in cuts]).encode()).hexdigest()
        # a jump cut inside one shot (an interview with a pause or a slip cut out) gets B-roll over it, as an editor hides it
        style_p = PROFILES.get(plan.get("style"), PROFILES["story"])
        jumps = [j for j, _ in jump_cuts(intervals, files, cuts)] if (not keep and not mont and not style_p["punch_ins"] and not ad
                                                                       and self.cfg.get("clipper.ai_videos.cover_jumps", True)) else []
        fresh = False
        if reuse and rc.get("csig") == csig and rc.get("final") is not None:
            cutaways, cuts = [dict(c) for c in rc["final"]["cutaways"]], [dict(c) for c in rc["final"]["cuts"]]
        else:
            cutaways, cuts = keep_moving(cutaways, cuts, intervals, pool if pool else [], allowed if pool and ad else None, group, out_len,
                                         exclude={outro["scene"]} if outro else set(), choose=choose, boundaries=step_boundaries,
                                         jumps=jumps, share=JUMP_SHARE.get(pace, 0.6))
            fresh = True
        if nopic:                                                           # (an older saved placement: cover what no camera filmed now)
            cutaways, _ = cover_spans(cutaways, [(out_time(intervals, a), out_time(intervals, b)) for a, b in nopic], pool,
                                      exclude={outro["scene"]} if outro else set())
        bad = AV.bad_pool(AV.load(st), pool or [], files)                   # shots the person ruled out ("Don't use this"): never kept, even in a saved placement
        if bad:
            cutaways, did = AV.swap(cutaways, pool or [], bad, exclude={outro["scene"]} if outro else set())
            if did["swapped"] or did["removed"]:
                log.info(f"B-roll ruled out by the editor: {did['swapped']} swapped for another shot, {did['removed']} taken out")
                NT.note("info", f"{did['swapped']} B-roll shot{'s' if did['swapped'] != 1 else ''} you ruled out swapped for another"
                        + (f", {did['removed']} taken out (nothing else fitted)" if did["removed"] else ""))
                if not fresh and rc.get("final") is not None:               # (a saved placement keeps the swap)
                    st.put("render_cache.json", {**rc, "final": {**rc["final"], "cutaways": cutaways}})
        if pool and cutaways and not mont and not ad and not (rc.get("pinned")):     # no file and no look more than twice (a shot swapped by hand stays as it is)
            cutaways, varied = MO.vary(cutaways, pool, exclude={outro["scene"]} if outro else set())
            if varied:
                log.info(f"B-roll variety: {varied} repeated shot(s) swapped for different ones")
        cutaways = no_flash_gaps(cutaways, pool or [], out_len)             # never a half-second flash of the speaker between two shots
        if fresh:                                                          # (an ad's too, so a shot can be swapped; it is only reused once pinned)
            st.put("render_cache.json", {"sig": rsig, "planned": planned, "csig": csig, "final": {"cutaways": cutaways, "cuts": cuts}})
        plan.pop("stock_credits", None)
        if any(f.get("stock") for f in files) or (st.get("stock.json") or {}).get("photos"):
            P("rendering", "Downloading the stock B-roll the edit uses", 0.595)
            plan["stock_credits"] = self.fetch_stock(cutaways, files, st)    # only the placed shots are downloaded
        self._live_broll(cutaways, pool or [], files)
        files = files + self.bake_photos(cutaways, files, st)               # photos become short animated clips (Final Cut gets the same clips)
        # the graphics in the folder (a logo, lower thirds, a title, see-through videos); "None" for graphics leaves them out too
        from . import foldergfx as FG
        fgx = self.graphics_of(folder, st, files) if opts.get("graphics", "title") else []
        logo = FG.logo_of(fgx) if fgx else None
        if logo and outro is None and not keep:                             # every video ends on your logo (an ad: unless it has its own end card)
            try:
                ref = next((f for f in files if not f["broll"]), files[0])
                CW, CH = (1920, 1080) if ref["w"] >= ref["h"] else (1080, 1920)
                lp = Path(logo)
                card = st.dir / f"endcard_{hashlib.sha1(f'{lp}:{lp.stat().st_size}:{int(lp.stat().st_mtime)}:{CW}x{CH}'.encode()).hexdigest()[:12]}.mp4"
                if not card.exists():
                    FG.logo_card(logo, CW, CH, OUTRO_S + 0.5, card)
                outro = {"path": str(card), "ss": 0.0, "d": OUTRO_S, "w": CW, "h": CH, "fps": 30.0, "tc": 0.0, "file_dur": OUTRO_S + 0.5, "card": True}
                if getattr(self, "live_job", None):
                    live.stage(self.store, self.live_job, f"End card: your logo ({lp.name})")
            except Exception as e:  # noqa: BLE001 - the video ends without it
                log.warning(f"the end card with your logo could not be made: {e}")
        # assemble from the ORIGINAL files (only the kept stretches are read), then finish
        P("rendering", "Cutting the kept stretches from the originals", 0.6)
        rough, rough_len, parts = self.assemble(files, intervals, cutaways, group, cuts, st,
                                                lambda f: P("rendering", f"Cutting the kept stretches ({f:.0%})", 0.6 + 0.12 * f), outro=outro,
                                                punch=not keep and not mont and bool(PROFILES.get(plan.get("style"), PROFILES["story"])["punch_ins"]),   # (a kept video is never zoomed)
                                                even=not keep and not mont,       # (...nor re-levelled: it stays exactly as it is)
                                                style=None if (keep or mont) else plan.get("style") or "story",   # (layouts: never in a kept video)
                                                tmode="none" if keep else ("full" if mont and mont.get("dissolves") else "none" if mont and TR.mode_of(opts.get("transitions")) == "none"
                                                                            else "auto") if mont else TR.mode_of(opts.get("transitions")),     # (a montage: hard cuts and an end fade; soft dissolves only when switched on)
                                                hook_at_start=bool(plan.get("hook_text")))
        if getattr(self, "live_job", None):                                  # the live view: frames of the cut as it now stands
            live.filmstrip(self.store, self.live_job, rough, rough_len, of="the cut", n=8, what="the cut")
        endcard = None
        if outro and not outro.get("card"):                                  # an ad's closing B-roll shot gets its name, call to action and logo
            end = self.end_info(ai, folder, plan, self.projects.brief(proj), st)
            endcard = {"at": round(rough_len - OUTRO_S, 2), "len": OUTRO_S, "text": " - ".join(x for x in (end["business"], end["tagline"]) if x),
                       "business": end["business"], "tagline": end["tagline"], "logo": end.get("logo")}
        # Final Cut gets a real timeline built from the ORIGINAL files for every project - a multicam clip when there is a shoot, plain
        # clips otherwise - so each cut can be extended, each B-roll shot moved; the sound is the same recording the video uses
        mc = {"group": group, "items": fcp_items(parts), "cutaways": cutaways_of_parts(parts, files),
              "transitions": list(getattr(self, "last_transitions", None) or [])}          # (what assemble decided: Final Cut gets the same ones)
        words_rough = remap_words(words, intervals, 1.0)
        jobd = self.store.dir(job)
        for f in jobd.glob("clip_*"):
            f.unlink(missing_ok=True)
        shutil.rmtree(jobd / "fcp", ignore_errors=True)
        points = []
        gmode = opts.get("graphics", "title")                                      # "" none | "title" the animated title card (+ key points) | "all" every kind of graphic
        scene_gfx: list[dict] = []
        if gmode == "all":                                                          # slides, charts, stats, quotes, lower thirds: chosen from what is SAID
            from . import gfxplan
            P("rendering", "Choosing the on-screen graphics from the transcript", 0.69)
            out_words = remap_words(words, intervals, 1.0)
            scene_gfx = gfxplan.plan(ai, seg(out_words), self.projects.brief(proj), PROFILES.get(plan.get("style"), PROFILES["story"]),
                                     rough_len - (OUTRO_S + 0.3 if outro else 0.0), hook_end=3.2 if plan.get("hook_text") else 0.0, words=out_words, cfg=self.cfg)
            plan["gfx_credits"] = sorted({i["credit"] for g in scene_gfx for i in (g.get("items") or []) if i.get("credit")})      # photos need their credit
            log.info(f"graphics: {len(scene_gfx)} planned ({', '.join(g['kind'] for g in scene_gfx)})")
            try:
                (jobd / "graphics_check.md").write_text(gfxplan.check_notes(scene_gfx, seg(out_words)))                   # what every graphic shows, and what you said for it
            except Exception as e:  # noqa: BLE001 - a list for a human is never a reason to fail a render
                log.warning(f"could not write the graphics check list: {e}")
        if gmode == "title" and (plan.get("style") in ("promo", "ad") or GRAPHIC_RE.search(proj.get("notes") or "")):
            P("rendering", "Choosing the key points for on-screen graphics", 0.7)
            points = key_points(ai, seg(remap_words(words, intervals, 1.0)), self.projects.brief(proj),
                                rough_len - (OUTRO_S + 0.3 if outro else 0.0))
        folder_gfx: list[dict] = []
        bug = None
        placed = {"title": False}
        plan.pop("folder_graphics", None)
        if fgx:                                                                     # your own graphics, each where it belongs
            prog_len = rough_len - (OUTRO_S if outro else 0.0)
            placed = FG.place(fgx, seg(remap_words(words, intervals, 1.0)), prog_len, proj.get("notes") or "", ai)
            folder_gfx = [{k: o.get(k) for k in ("name", "path", "kind", "role", "label", "at", "len", "w", "h", "dur")} for o in placed["overlays"]]
            if folder_gfx:                                                          # made-up graphics never cover yours
                def clash(a: float, ln: float) -> bool:
                    return any(a < o["at"] + o["len"] + 0.1 and o["at"] < a + ln + 0.1 for o in folder_gfx)
                points = [c for c in points if not clash(float(c["at"]), float(c["len"]))]
                scene_gfx = [c for c in scene_gfx if not clash(float(c["at"]), float(c["len"]))]
            if placed["bug"]:                                                       # a corner logo the whole time: only when the notes ask
                t0 = max([o["at"] + o["len"] for o in folder_gfx if o["role"] == "title"] + [0.0])
                if prog_len - t0 > 2.0:
                    bug = {"logo": placed["logo"], "at": round(t0, 2), "len": round(prog_len - t0, 2)}
            plan["folder_graphics"] = [{"name": o["name"], "role": o["role"], "at": o["at"], "len": o["len"]} for o in folder_gfx] + (
                [{"name": Path(logo).name, "role": "end card", "at": round(prog_len, 2), "len": OUTRO_S}] if logo and outro and outro.get("card") else [])
            if getattr(self, "live_job", None):
                for o in plan["folder_graphics"]:
                    live.stage(self.store, self.live_job, f"Your graphic {o['name']} ({o['role']}) at {o['at']:.1f} s")
        loud = (float(self.cfg.get("clipper.ai_videos.loudness_lufs_promo", -12.0)) if plan.get("style") in ("promo", "ad")
                else float(self.cfg.get("clipper.ai_videos.loudness_lufs", -14.0)))                  # an ad / promo finishes louder than a story
        jopts = {**opts, "mode": "polish", "project": True, "fcp": True, "hook_text": plan.get("hook_text", ""),
                 "callouts": points, "endcard": endcard, "gfx_theme": plan.get("theme"), "music_file": str(track.path) if track else "none", "music_start": music_start,
                 "music_vibe": {"mood": (plan.get("music") or {}).get("mood", ""), "genre": (plan.get("music") or {}).get("genre", ""),
                                "why": (plan.get("music") or {}).get("why", "chosen"), "by": "ai"},
                 "gfx_cards": scene_gfx, "graphics": gmode if (plan.get("hook_text") or scene_gfx) else "", "loudness_lufs": loud,
                 "folder_gfx": folder_gfx, "folder_title": bool(placed["title"]), "logo_bug": bug}
        if placed["title"]:                                                        # your own title graphic opens the video: no fade in from black under it
            mc["transitions"] = [e for e in mc["transitions"] if e["kind"] != "fade_in"]
        jopts["fades"] = TR.fade_list(mc["transitions"])                           # (the fades at the ends, made in the final picture by polish.run)
        self.store.update(job, options=jopts, title=plan["title"], info={})
        from . import polish
        P("rendering", "Captions, music and the title card", 0.74)
        try:
            try:
                clip = polish.run(engine, job, rough, words_rough, override=[(0.0, rough_len)], mc=mc)
            except ffmpeg.FFmpegError as e:                                 # a stalled / failed final render: once more, without the graphics
                log.warning(f"the final render failed ({str(e)[:200]}); trying again without the on-screen graphics")
                self.store.update(job, options={**jopts, "no_overlays": True})
                clip = polish.run(engine, job, rough, words_rough, override=[(0.0, rough_len)], mc=mc)
        finally:
            self.store.update(job, options=opts)                            # polish's working settings must never replace what you chose
        loudness_pass(jobd / clip["file"], loud)
        self.verify(jobd / clip["file"], rough_len, reference=rough)
        outline = FL.steps(proj.get("notes") or "") if not keep and not mont else []         # (a kept video says what it says: no script to check)
        if outline:
            # The final arbiter listens to the exported media, not the intended plan. If it cannot quote proof,
            # the corresponding script point is explicitly marked missing for the creator's review.
            P("rendering", "Verifying the finished render against your script", 0.92)
            try:
                rendered_words = self._words_of({"path": str(jobd / clip["file"]), "dur": rough_len}, ai, opts.get("transcriber", "parakeet"))
                rendered = FL.final_check(ai, seg(rendered_words), outline)
            except Exception as e:  # noqa: BLE001 - never pretend a failed verification passed
                log.warning(f"final rendered-script verification failed: {e}")
                rendered = [{"n": n, "step": step, "missing": [f"The finished render could not be verified: {type(e).__name__}"],
                             "quotes": [], "rendered": True} for n, step in enumerate(outline, 1)]
            final_by_step = {x["n"]: x for x in rendered}
            plan["final_check"] = rendered
            if plan.get("flow"):
                plan["flow"] = [{**row, "missing": final_by_step.get(row["n"], {}).get("missing", row.get("missing", [])),
                                 "quotes": final_by_step.get(row["n"], {}).get("quotes", []), "final_rendered": True,
                                 "covered": not final_by_step.get(row["n"], {}).get("missing", row.get("missing", []))}
                                for row in plan["flow"]]
        version = int(plan.get("version", 0)) + 1
        plan["version"] = version
        plan["history"] = [*plan.get("history", []), {"n": version, "note": note, "at": int(time.time()), "beat": beat_note}]
        plan.pop("pending_render", None)
        plan.pop("pending_note", None)
        plan.pop("pending_changes", None)                                   # (the B-roll changes waiting on the strip are in this render)
        previous_plan = st.get("plan.json")
        if previous_plan:
            st.snapshot("plan.json", previous_plan, "before-render-result")
        st.put("plan.json", plan)
        footage = sum(float(f.get("dur") or 0) for f in files if not f.get("photo") and not f.get("broll"))
        clip = {**clip, "story": {"summary": plan["summary"], "files": []}, "cutaways": cutaways, "hook_text": plan.get("hook_text"),
                "beat": beat_note, "cost_usd": ai.cost(), "local": True, "cuts": max(0, len(intervals) - 1), "footage_s": round(footage, 1),
                "trimmed_s": round(max(0.0, footage - rough_len), 1),
                "multicam": ({"angles": [a["id"] for a in group["angles"]], "recorders": len(group["recorders"]), "switches": len(cuts)} if group else None)}
        self.store.set_clip(job, clip)
        (jobd / "plan.json").write_text(json.dumps({"intervals": [[round(a, 2), round(b, 2)] for a, b in intervals], "cutaways": cutaways,
                                                   "angles": cuts}))
        P("rendering", "Saving into the folder", 0.97)
        self.deliver(pid, plan, clip, folder, words, version)                # only reached once the video passed its checks
        spent = ai.cost()
        self.projects.update(pid, cost_usd=round(float(self.projects.get(pid).get("cost_usd", 0)) + spent, 4))
        self.store.update(job, status="done", progress=1.0, error=None,
                          stage=f"Done - v{version} saved in the folder (this run ~${spent:.2f})" + script_note(plan.get("flow")))
        self.projects.finished(pid)

    # -- how good the sound of what is kept is (soundcheck.py) ------------------------------------------------------------
    def grades(self, st: State):
        """The project's sound grades (10-second windows of each recording), loaded once per run; save_grades() writes them back."""
        from . import soundcheck
        g = getattr(self, "_grades", None)
        if g is None or getattr(self, "_grades_dir", None) != st.dir:
            g = soundcheck.Grades(dict(st.get("sound_grades.json") or {}))
            self._grades, self._grades_dir = g, st.dir
        return g

    def clean_voices(self, parts: list[dict], st: State) -> list[dict]:
        """Each stretch of one recording whose sound is weak or bad (soundcheck grades; or every one with "Always") is cleaned once,
        as a whole (voiceclean.py: no seam at the cuts inside it). Its pieces then read the cleaned copy (clean_path / clean_ss): only
        the cutter does, so the Final Cut project keeps the original recordings. Settings › How it edits › Clean up voices; nothing
        happens when cleanup isn't installed or is set to Never."""
        from . import voiceclean as VC
        if not VC.available(self.cfg) or VC.mode("long", self.cfg) == "never":
            return parts
        g = self.grades(st)
        done = []
        for r in sound_runs(parts):
            a = parts[r[0]].get("a")
            if not a or not a.get("path"):
                continue
            ss0 = max(0.0, float(a["ss"]))
            d = sum(float(parts[i]["d"]) for i in r)
            grade = g.span(a["path"], ss0, ss0 + d)["grade"]
            if not VC.wanted(grade, "long", self.cfg):
                continue
            try:
                wav, start = VC.cleaned_range(a["path"], ss0, ss0 + d, st.dir / "clean", cfg=self.cfg)
            except Exception as e:  # noqa: BLE001 - the original sound is used instead
                log.warning(f"voice cleanup skipped for {Path(a['path']).name}: {e}")
                continue
            for i in r:
                pa = parts[i].get("a")
                if pa and pa.get("path") == a["path"]:
                    parts[i]["a"] = {**pa, "clean_path": str(wav), "clean_ss": float(pa["ss"]) - start}
            done.append({"file": Path(a["path"]).name, "from": round(ss0, 1), "to": round(ss0 + d, 1), "grade": grade})
        self.save_grades(st)
        if done:
            st.put("voice_clean.json", {"cleaned": done, "at": time.time()})        # (Notes can say which stretches were cleaned)
            log.info(f"voice cleanup: {len(done)} stretch(es) of {len({x['file'] for x in done})} recording(s) cleaned")
        return parts

    def save_grades(self, st: State) -> None:
        g = getattr(self, "_grades", None)
        if g is not None and g.dirty and getattr(self, "_grades_dir", None) == st.dir:
            with g.lock:
                st.put("sound_grades.json", dict(g.data))
                g.dirty = False

    def sound_of(self, files: list[dict], group: dict | None, st: State, a: float, b: float) -> dict | None:
        """The grade of the sound heard for program seconds a..b (None: nothing to judge, e.g. a silent file)."""
        if not self.cfg.get("clipper.ai_videos.measure_sound", True):
            return None
        mid = (a + b) / 2
        src = sound_source(files, group, mid)
        if not src:
            return None
        path, loc = src
        return self.grades(st).span(path, max(0.0, loc - (mid - a)), loc + (b - mid))

    def mark_sound(self, cands: list[dict], files: list[dict], st: State) -> list[dict]:
        """Candidate moments whose sound is poor say so in what the assembler reads ("[poor sound: muffled]"), and score lower (bad -3,
        weak -1): good lines are kept first, a badly recorded one only when it matters (a workshop shoot opened on C0018's camera mic)."""
        group = (st.get("multicam.json") or {}).get("group")
        out = []
        for c in cands:
            try:
                g = self.sound_of(files, group, st, float(c["start"]), float(c["end"]))
            except Exception:  # noqa: BLE001
                g = None
            if g and g["grade"] in ("bad", "weak"):
                c = {**c, "sound": g["grade"], "score": max(1.0, float(c["score"]) - (3.0 if g["grade"] == "bad" else 1.0)),
                     "what": f"[{'poor' if g['grade'] == 'bad' else 'so-so'} sound{': ' + g['why'] if g.get('why') else ''}] {c['what']}"}
            out.append(c)
        self.save_grades(st)
        n = sum(1 for c in out if c.get("sound") == "bad")
        if n:
            log.info(f"sound: {n} of {len(out)} candidate moments were recorded badly (marked for the editor)")
        return out

    def rate_sound(self, st: State) -> None:
        mc = st.get("multicam.json") or {}
        group = mc.get("group")
        if not group or any(c["path"] not in (group.get("sound_q") or {}) for a in group["angles"] for c in a["files"]):
            if group:
                MC.rate_sound(group, TI.Meter(st.dir))
                st.put("multicam.json", {**mc, "group": group})
                log.info("camera sound quality: " + ", ".join(f"{Path(p).name} {v}" for p, v in group["sound_q"].items()))
        group = (st.get("multicam.json") or {}).get("group")
        if group and self.cfg.get("clipper.ai_videos.measure_sound", True) and any(
                c["path"] not in (group.get("sound_r") or {}) for c in [c for a in group["angles"] for c in a["files"]] + list(group.get("recorders") or [])):
            MC.rate_sources(group)                                       # recorders too: the cleanest recording is heard, not always the recorder
            st.put("multicam.json", {**(st.get("multicam.json") or {}), "group": group})
            log.info("sound by measurement: " + ", ".join(f"{Path(p).name} {v['score']} ({v['grade']})" for p, v in group["sound_r"].items()))

    def bake_photos(self, cutaways: list[dict], files: list[dict], st: State) -> list[dict]:
        """Each photo cutaway becomes a short silent clip with its effect; returns the extra 'files' that stand for them."""
        out, k = [], 0
        ref = next((f for f in files if not f["broll"]), files[0])
        W, H = (1920, 1080) if ref["w"] >= ref["h"] else (1080, 1920)
        pdir = st.dir / "photos"
        pdir.mkdir(exist_ok=True)
        for c in cutaways:
            if not c.get("photo"):
                continue
            effect = PH.EFFECTS[k % len(PH.EFFECTS)]
            k += 1
            secs = float(c["length"]) + 0.5
            mov = pdir / f"{PH.key(c['photo'], effect, secs, W, H)}.mp4"
            try:
                if not mov.exists():
                    PH.bake(c["photo"], effect, secs, W, H, 30.0, mov)
            except Exception as e:  # noqa: BLE001 - this cutaway just stays on the speaker
                log.warning(f"animating {Path(c['photo']).name} failed: {e}")
                c["length"] = 0.0
                continue
            out.append({"name": f"photo {Path(c['photo']).name}", "path": str(mov), "start": c["src"], "end": c["src"] + secs, "dur": secs, "w": W, "h": H,
                        "fps": 30.0, "audio": False, "broll": True, "photo": True, "tc": 0.0})
        cutaways[:] = [c for c in cutaways if c["length"] > 0.3]
        return out

    def tighten(self, intervals, files: list[dict], st: State, bounds: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
        """Move every cut to where the sound really stops or starts (never mid-word, no dead air), and keep the interviewer out. `bounds`
        (one per stretch) are the limits from the transcript: a cut never runs on into the next word or back into the one before."""
        group = (st.get("multicam.json") or {}).get("group")

        def source_at(t: float):
            f = next((x for x in files if x["start"] <= t < x["end"] + 1e-6), None)
            if f is None:
                return None
            loc = t - f["start"]
            if f.get("multicam") and group:
                s = MC.sound_at(group, loc)
                return (s["path"], s["local"]) if s else None
            if f.get("clean"):
                return (f["clean"]["audio"], loc + float(f["clean"]["offset"]))
            return (f["path"], loc) if f.get("audio") else None
        spans = [tuple(x) for x in (st.get("roles.json") or {}).get("spans", [])]
        try:
            out = TI.refine([tuple(i) for i in intervals], source_at, TI.Meter(st.dir), spans, bounds=bounds)
        except Exception as e:  # noqa: BLE001 - the original cut points are always usable
            log.warning(f"tightening the cuts failed ({e}); keeping the transcript's cut points")
            out = [tuple(i) for i in intervals]
        return TI.without(out, spans) if spans else out

    def direct(self, group: dict, intervals, ai: LocalAI, words: list[dict], brief: str, st: State) -> list[dict]:
        """The director's angle choices for the kept stretches of the shoot (cached for the same cut, so a re-render doesn't change them)."""
        stretches = [(max(0.0, a), min(b, group["dur"])) for a, b in intervals if a < group["dur"] and min(b, group["dur"]) - max(0.0, a) > 0.2]
        key = hashlib.sha1(json.dumps([[round(a, 1), round(b, 1)] for a, b in stretches] + [a["id"] for a in group["angles"]]).encode()).hexdigest()
        c = st.get("cuts.json", {})
        if c.get("key") == key:
            return c["cuts"]
        cuts = MC.choose_angles(ai, group, stretches, brief, words)
        st.put("cuts.json", {"key": key, "cuts": cuts})
        return cuts

    def verify(self, path: Path, expected: float, reference: Path | None = None) -> None:
        """The finished video must be whole before anyone sees it: picture and sound the same length, the length we planned, and no
        stretch where the picture stops while the sound plays on. A video that fails is not delivered. (A still that is still in the
        rough cut too - a photo, a static screen - is the footage, not a fault.)"""
        problems = check_video(path, expected, bool(self.cfg.get("clipper.ai_videos.verify_freeze", True)), reference=reference)
        if problems:
            raise RuntimeError("the finished video failed its check, so it was NOT saved into your folder: " + "; ".join(problems))

    def assemble(self, files: list[dict], intervals, cutaways, group, cuts, st: State, progress, outro: dict | None = None,
                 punch: bool = False, even: bool = True, style: str | None = None, tmode: str = "none",
                 hook_at_start: bool = False) -> tuple[Path, float, list[dict]]:
        """Cut ONLY the kept stretches out of the original files and join them: a small rough cut, however big the footage is.
        tmode "auto": transitions where the cut calls for one (transitions.py): dissolves are made here, the fades at the ends in polish.run;
        what was decided is kept in self.last_transitions for the Final Cut project.
        Picture and sound are chosen separately for every piece: the speaker's sound runs through, the picture can come from another
        camera angle (multicam) or from B-roll (a cutaway). Returns (rough cut, its length, the pieces)."""
        talk = [f for f in files if not f["broll"] and not f.get("multicam")] or files
        ref = (MC.angle_files(group, max(group["angles"], key=lambda a: sum(c["dur"] for c in a["files"]))["id"])[0] if group else talk[0])
        W0, H0 = ref["w"], ref["h"]
        s = min(1.0, MAX_EDGE / max(W0, H0))
        W, H = int(W0 * s) // 2 * 2, int(H0 * s) // 2 * 2
        rate = timeline_rate(ref["fps"])                                       # the camera's own rate: 23.976 stays 23.976 (not 24)
        choose_layouts(cutaways, W, H, style, on=bool(self.cfg.get("clipper.ai_videos.layouts", True)) and style is not None)
        parts = build_parts(files, group, cuts, intervals, cutaways, outro)
        durs = picture_durations(files, group, outro)
        bad = bad_pictures(parts, durs)
        if bad:                                                               # a camera that wasn't rolling, sound with nothing over it
            raise RuntimeError("the cut would show a picture that doesn't exist (" + "; ".join(bad[:3]) + "), so it was stopped before rendering")
        parts, flashes = fix_flashes(smooth_angles(parts, group), durs)
        if flashes:
            log.warning(f"shots under a second are left at {flashes[:8]} s (nothing could run on over them)")
            for t in flashes[:6]:
                NT.note("warn", "a shot under a second that couldn't be covered", start=float(t), end=float(t) + 1.0,
                        do="Extend the shot before or after over it in Final Cut, or trim it out.")
        parts = seal_joins(quantize(parts, rate))
        if even and self.cfg.get("clipper.ai_videos.measure_sound", True):    # each stretch's speech at the same level (no 15 dB jumps)
            g = self.grades(st)
            runs = sound_runs(parts)                                          # one gain per stretch of one recording (never a jump inside it)
            lv = [g.span(parts[r[0]]["a"]["path"], max(0.0, float(parts[r[0]]["a"]["ss"])),
                         max(0.0, float(parts[r[0]]["a"]["ss"])) + sum(parts[i]["d"] for i in r))["level"] if parts[r[0]].get("a") else -99.0
                  for r in runs]
            for r, gain in zip(runs, level_gains(lv, [sum(parts[i]["d"] for i in r) for r in runs])):
                for i in r:
                    if gain:
                        parts[i]["gain"] = gain
            self.save_grades(st)
        if even:                                                              # noisy, far-away speech cleaned (voiceclean.py), MP4 only
            parts = self.clean_voices(parts, st)
        if punch and self.cfg.get("clipper.ai_videos.punch_ins", True):          # only for kinds of video that are meant to feel energetic (PROFILES)
            parts = punch_ins(parts, float(self.cfg.get("clipper.ai_videos.punch_in_zoom", 1.12)))
        for k, p in enumerate(parts):                                         # a tiny fade wherever the sound jumps to another place (no click)
            prv, nxt = (parts[k - 1] if k else None), (parts[k + 1] if k + 1 < len(parts) else None)
            p["fade_in"] = not (prv and joined(prv, p))
            p["fade_out"] = not (nxt and joined(p, nxt))
        try:                                                                  # where a transition belongs (never a reason to fail a render)
            events = TR.plan(parts, rate, style=style or "story", mode=tmode, hook_at_start=hook_at_start, durs=durs, outro=bool(outro))
        except Exception as e:  # noqa: BLE001
            log.warning(f"transitions left out: {e}")
            events = []
        self.last_transitions = events
        cut_parts = TR.render_parts(parts, events, rate) if any(e["kind"] in ("dissolve", "broll") for e in events) else parts
        if events:
            log.info("transitions: " + ", ".join(f"{e['kind']} at {e['at']:.1f}s" for e in events))
        seg_dir = st.dir / "seg"
        shutil.rmtree(seg_dir, ignore_errors=True)
        seg_dir.mkdir()
        rs = f"{rate.numerator}/{rate.denominator}"
        norm = f"fps={rs},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
        fill = f"fps={rs},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,format=yuv420p"
        masks = layout_masks(seg_dir, W, H) if any(p["v"].get("layout") == "pip" for p in parts) else None
        done = [0]
        lock = threading.Lock()
        ffb = ffmpeg.bin_path("ffmpeg")

        def cut(i_p):
            """One piece, EXACTLY its frame count of picture and the matching count of sound samples, so the joined rough cut has no
            holes (a piece one frame short used to leave a one-frame hitch at the cut) and never drifts out of sync."""
            i, p = i_p
            if p.get("xf"):                                                 # a transition: two pieces carried past their edges, blended
                return blend(i, p)
            out = seg_dir / (f"{i:05d}.mov" if isinstance(i, int) else f"{i}.mov")
            n = int(p["n"])
            d = float(Fraction(n) / rate)
            samples = int(round(Fraction(n * 48000) / rate))
            v, a = p["v"], p["a"]
            ff = 0.008
            fades = (f",afade=t=in:st=0:d={ff}" if p.get("fade_in") else "") + (f",afade=t=out:st={max(0.0, d - ff):.3f}:d={ff}" if p.get("fade_out") and d > 0.1 else "")
            gain = f"volume={p['gain']}dB," if p.get("gain") else ""             # (level_gains: the same speech level everywhere)
            tail = f"{gain}aresample=48000,aformat=channel_layouts=stereo,apad,atrim=end_sample={samples},asetpts=PTS-STARTPTS{fades}[a]"
            if a:                                                           # input 1: the sound (a clean recording, or the camera's own)
                if a.get("clean_path"):                                     # (the cleaned copy of this recording: the MP4 only)
                    a = {**a, "path": a["clean_path"], "ss": a["clean_ss"]}
                pre = max(0.0, -a["ss"])
                sound = ["-ss", f"{max(0.0, a['ss']):.6f}", "-t", f"{max(0.1, d - pre + 0.2):.3f}", "-i", a["path"]]
                af = "[1:a:0]" + (f"adelay={int(pre * 1000)}|{int(pre * 1000)}," if pre > 0 else "") + tail
            else:
                sound = ["-f", "lavfi", "-t", f"{d + 0.2:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                af = "[1:a]" + tail

            def picture(src: dict, chain: str, careful: bool) -> list[str]:
                """input 0 + the whole filter graph. careful: seek to 10 s before and trim to the frame (slower, but works where a
                fast seek doesn't); the seek always applies to the PICTURE input."""
                back = min(10.0, src["ss"]) if careful else 0.0
                head = f"trim=start={back:.3f},setpts=PTS-STARTPTS," if careful else ""
                zoom = (f",scale=trunc({W}*{p['zoom']}/2)*2:trunc({H}*{p['zoom']}/2)*2,crop={W}:{H}" if p.get("zoom") and not src.get("cut") else "")
                lut = LC.vf_prefix(src["path"], self.cfg)               # log footage (S-Log3...) -> normal colour in the MP4 (not in Final Cut)
                vf = f"[0:v:0]{lut}{head}{chain}{zoom},tpad=stop_mode=clone:stop_duration=2,trim=end_frame={n},setpts=PTS-STARTPTS[v]"
                return ["-ss", f"{src['ss'] - back:.3f}", "-t", f"{d + back + 0.5:.3f}", "-i", src["path"], *sound,
                        "-filter_complex", f"{vf};{af}"]
            enc = ["-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                   "-c:a", "pcm_s16le", str(out)]
            inset = p.get("v0") if v.get("layout") and v.get("cut") else None
            if inset and inset.get("path") and not inset.get("nopic"):         # a layout: the B-roll AND the speaker's own picture
                try:
                    lay = v["layout"]
                    fy = face_y(inset["path"], inset["ss"], d, self.cfg.root / "data" / "models") if lay == "split" else FACE_Y
                    g = layout_graph(lay, W, H, rs, LC.vf_prefix(v["path"], self.cfg), LC.vf_prefix(inset["path"], self.cfg), n, masks, fy)
                    still = (["-loop", "1", "-t", f"{d + 0.5:.3f}", "-i", str(masks["mask"]), "-loop", "1", "-t", f"{d + 0.5:.3f}", "-i", str(masks["back"])]
                             if lay == "pip" else [])
                    ffmpeg.run(["nice", "-n", "15", ffb, "-y", "-v", "error",
                                "-ss", f"{v['ss']:.3f}", "-t", f"{d + 0.5:.3f}", "-i", v["path"], *sound,
                                "-ss", f"{float(inset['ss']):.3f}", "-t", f"{d + 0.5:.3f}", "-i", inset["path"], *still,
                                "-filter_complex", f"{g};{af}", *enc], timeout=3600, what=f"cutting a stretch ({lay})")
                    with lock:
                        done[0] += 1
                        progress(min(1.0, done[0] / ncut))
                    return out
                except Exception as e:  # noqa: BLE001 - the plain B-roll shot is always there to fall back on
                    log.warning(f"a {v.get('layout')} layout failed ({str(e)[:160]}); showing the B-roll full screen there")
            try:
                ffmpeg.run(["nice", "-n", "15", ffb, "-y", "-v", "error", *picture(v, fill if v["fill"] else norm, False), *enc],
                           timeout=3600, what="cutting a stretch")
            except ffmpeg.FFmpegError as e:
                log.warning(f"a cut failed ({str(e)[:160]}); retrying with accurate seeking")
                try:
                    ffmpeg.run(["nice", "-n", "10", ffb, "-y", "-v", "error", *picture(v, fill if v["fill"] else norm, True), *enc],
                               timeout=3600, what="cutting a stretch (careful)")
                except ffmpeg.FFmpegError:
                    v0 = p.get("v0")
                    if not v0:
                        raise
                    log.warning("the B-roll for this cutaway couldn't be read; using the speaker's picture")
                    ffmpeg.run(["nice", "-n", "10", ffb, "-y", "-v", "error", *picture(v0, norm, False), *enc],
                               timeout=3600, what="cutting a stretch (speaker)")
            with lock:
                done[0] += 1
                progress(min(1.0, done[0] / ncut))
            return out

        def blend(i, p):
            """A transition (transitions.py): two pieces of 2h frames, each carried past its own edge, cross-dissolved over their whole length;
            the sound crossfades (a different place) or just carries on (the same recording under a B-roll shot). Same frame count as the two
            halves it replaces, so nothing after it moves."""
            x = p["xf"]
            fa, fb = cut((f"{i:05d}a", x["a"])), cut((f"{i:05d}b", x["b"]))
            out = seg_dir / f"{i:05d}.mov"
            n = int(p["n"])
            dur = float(Fraction(n) / rate)
            samples = int(round(Fraction(n * 48000) / rate))
            audio = (f"[0:a][1:a]acrossfade=d={dur:.6f}:c1=qsin:c2=qsin" if x.get("audio") == "crossfade" else "[0:a]anull")
            g = (f"[0:v][1:v]xfade=transition=fade:duration={dur:.6f}:offset=0,fps={rs},trim=end_frame={n},setpts=PTS-STARTPTS,format=yuv420p[v];"
                 f"{audio},apad,atrim=end_sample={samples},asetpts=PTS-STARTPTS[a]")
            ffmpeg.run(["nice", "-n", "15", ffb, "-y", "-v", "error", "-i", str(fa), "-i", str(fb), "-filter_complex", g, "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(out)],
                       timeout=3600, what="a transition")
            fa.unlink(missing_ok=True)
            fb.unlink(missing_ok=True)
            return out
        ncut = len(cut_parts) + 2 * sum(1 for q in cut_parts if q.get("xf"))
        with ThreadPoolExecutor(max_workers=2) as ex:                       # two at a time: your Mac stays usable while it works
            outs = list(ex.map(cut, enumerate(cut_parts)))
        for o in outs:
            try:
                has_pic = ffmpeg.video_stream(ffmpeg.probe(o)) is not None
            except Exception:  # noqa: BLE001
                has_pic = False
            if not has_pic:
                raise RuntimeError(f"a cut came out without any picture ({o.name}); the source file may be damaged near that spot")
        lst = st.dir / "concat.txt"
        lst.write_text("".join(f"file '{o.as_posix()}'\n" for o in outs))
        for old in st.dir.glob("rough*.mov"):
            old.unlink(missing_ok=True)
        rough = st.dir / f"rough_{int(time.time())}.mov"
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(rough)],
                   timeout=3600, what="joining the stretches")
        shutil.rmtree(seg_dir, ignore_errors=True)
        return rough, ffmpeg.duration(rough), parts

    def end_info(self, ai, folder: Path, plan: dict, brief: str, st: State) -> dict:
        """The end card of an ad: the business name, a short call to action, and which picture in the project is the logo."""
        if plan.get("end"):
            return plan["end"]
        imgs = scan_graphics(folder)
        pics = []
        try:
            from io import BytesIO

            from PIL import Image
            for g in imgs[:4]:
                with Image.open(g["path"]) as im:
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    bg.alpha_composite(im)
                    im = bg.convert("RGB")
                    im.thumbnail((512, 512))
                    buf = BytesIO()
                    im.save(buf, "JPEG", quality=80)
                    pics.append(buf.getvalue())
        except Exception as e:  # noqa: BLE001 - no pictures to look at: the card is text only
            log.warning(f"looking at the project's graphics failed: {e}")
            pics = []
        schema = {"type": "object", "properties": {"business": {"type": "string"}, "tagline": {"type": "string"}, "logo_index": {"type": "integer"}},
                  "required": ["business", "tagline", "logo_index"]}
        prompt = (f"We are making an advertisement. The creator's notes: {brief[:1200]}\nWhat the ad says: {plan.get('summary', '')}\nTitle: {plan.get('title', '')}\n"
                  + (f"The project has {len(pics)} picture(s), shown in this order: " + ", ".join(g["name"] for g in imgs[:len(pics)]) + ". " if pics else "")
                  + "Give: 'business' = the business's name as it should appear on screen; 'tagline' = the closing call to action, 2 to 5 words "
                    "(for example 'Find Your Perfect Fit' or 'Visit Us Today'); 'logo_index' = the 0-based number of the picture that is the business's "
                    "LOGO (a brand mark or wordmark: flat shapes or lettering, NOT a photograph, thumbnail, screenshot or video frame), or -1 if none is. "
                    "When in doubt, -1.")
        info = {"business": "", "tagline": "Visit us today", "logo": None}
        try:
            res = ai.complete_json("ad_end", "You prepare the end card of a short video advertisement.", prompt, schema, images=pics or None)
            li = int(res.data.get("logo_index", -1))
            if 0 <= li < len(imgs) and not looks_like_logo(imgs[li]["path"]):
                log.info(f"{Path(imgs[li]['path']).name} is a photo, not a logo: the end card is text only")
                li = -1
            info = {"business": " ".join(str(res.data.get("business", "")).split())[:40], "tagline": " ".join(str(res.data.get("tagline", "")).split())[:48],
                    "logo": imgs[li]["path"] if 0 <= li < len(imgs) and li < len(pics) else None}
        except Exception as e:  # noqa: BLE001
            log.warning(f"preparing the end card failed: {e}")
        plan["end"] = info
        st.put("plan.json", plan)
        return info

    def deliver(self, pid: str, plan: dict, clip: dict, folder: Path, words: list[dict], version: int) -> None:
        """Copy the results INTO the folder: Output/, Final Cut/, Notes/ (the "save to" folder's <title>/ when one was chosen)."""
        proj = self.projects.get(pid)
        folder = home_of(proj)
        job = self.store.get(proj["job"])
        d = self.store.dir(proj["job"])
        title = safe(plan["title"], 70)
        if proj.get("home"):                                                 # a project folder of its own: links to the footage it was made from
            from .studio import link_raw
            try:
                link_raw(folder, [f["path"] for f in (State(folder).get("files.json") or []) if f.get("path") and not f.get("library") and not f.get("photo")])
            except OSError as e:
                log.warning(f"linking the raw footage failed: {e}")
        out = folder / "Output"
        out.mkdir(exist_ok=True)
        tmp = out / f".{title} v{version}.mp4.part"
        shutil.copy2(d / clip["file"], tmp)
        tmp.replace(out / f"{title} v{version}.mp4")                         # appears in one piece, finished
        if clip.get("srt") and (d / clip["srt"]).exists():
            shutil.copy2(d / clip["srt"], out / f"{title} v{version}.srt")
        fcp = clip.get("fcp")
        if fcp and (d / "fcp_project.zip").exists():
            target = folder / "Final Cut"
            target.mkdir(exist_ok=True)
            here = target / f"{fcp['folder']} v{version}"                    # every version its own folder: a project you already imported keeps its files
            shutil.rmtree(here, ignore_errors=True)
            with tempfile.TemporaryDirectory(prefix="fcp-", dir=target) as tmp:
                with zipfile.ZipFile(d / "fcp_project.zip") as z:
                    z.extractall(tmp)
                Path(tmp, fcp["folder"]).replace(here)
            xml = here / fcp["fcpxml"]
            if xml.is_file():                                        # point the project at this folder
                xml.write_text(xml.read_text().replace("file://" + quote(fcp["unpack_to"]), "file://" + quote(str(here.resolve()))))
                shutil.copy2(xml, folder / f"{title} v{version}.fcpxml")        # also right in the folder you gave: import this one in Final Cut
        older_versions_to_trash(folder, version)                           # only the newest video and .fcpxml stay in the folder
        notes = folder / "Notes"
        notes.mkdir(exist_ok=True)
        shutil.copy2(d / "plan.json", notes / "plan.json") if (d / "plan.json").exists() else None
        (notes / "plan.md").write_text(self.projects.plan_md({**proj, "version": version}, job, clip, d)
                                       + (f"\n## Hook text\n{plan.get('hook_text')}\n" if plan.get("hook_text") else "")
                                       + (f"\n## Beat\n{clip.get('beat')}\n" if clip.get("beat") else ""))
        (notes / "transcript.txt").write_text(" ".join(w["text"] for w in words) + "\n")
        if (d / "graphics_check.md").exists():
            shutil.copy2(d / "graphics_check.md", notes / "graphics_check.md")
        try:                                                                 # a check of the FINISHED video: dead audio, a lifeless first half, wrong length
            from . import qa as QA
            cams = {}
            try:
                g = json.loads((folder / ".aivideo" / "multicam.json").read_text()).get("group") or {}
                cams = {c["name"]: c["path"] for a in g.get("angles", []) for c in a.get("files", [])}
            except (OSError, ValueError):
                pass
            report = QA.check(out / f"{title} v{version}.mp4", plan.get("target_s"), cams)
            card = next((g for g in plan.get("folder_graphics") or [] if g.get("role") == "end card"), None)
            if card:                                                        # the logo end card is silent by design (no music): not dead audio
                report["issues"] = [i for i in report["issues"] if not (
                    (m := re.match(r"dead audio from ([\d.]+) s", str(i.get("what") or ""))) and float(m.group(1)) >= float(card["at"]) - 1.0)]
            from . import clipgate                                          # shared with the Clipper: a cut that ends mid-sentence or on a sign-off
            tail = " ".join(w["text"] for w in words)[-400:]                # (the words of the finished video: transcript.txt is written from the same list)
            for why in clipgate.problems({"duration": report.get("duration"), "transcript": tail}):
                if "seconds long" not in why:
                    report["issues"].append({"level": "warn", "what": f"the end of the video: {why}"})
            issues = report["issues"] + NT.take() + NT.sources(State(folder).get("files.json") or []) + self._log_note(folder)
            (notes / "quality.md").unlink(missing_ok=True)                  # (its old name)
            (notes / "Quality check.md").write_text(NT.quality_md(issues, report, f"{title} v{version}"))
            qa_counts = NT.counts(issues)
            qa_rows = NT.issue_rows(issues)[:60]
            for i in report["issues"]:
                if i["level"] != "info":
                    log.warning(f"quality check ({title} v{version}): {i['what']}")
        except Exception as e:  # noqa: BLE001 - the check is a bonus, never a reason to fail a render
            log.warning(f"quality check failed: {e}")
            qa_counts, qa_rows = None, []
        if plan.get("flow"):
            (notes / "flow.md").write_text("# Your script, checked point by point\n\n" + "\n".join(
                f"{r['n']}. {r['step']}\n   " + ("TOLD: " + r["text"] if r.get("text") else
                                                ("TOLD in the changed cut" if r.get("rechecked") and r.get("covered") else "NOTHING USED FOR THIS STEP"))
                + (f"\n   NOT TOLD (the cut does not clearly say): {', '.join(r['missing'])}" if r.get("missing") else "")
                + (f"\n   NOT IN THE FOOTAGE (the guest never says it): {', '.join(r['not_in_footage'])}" if r.get("not_in_footage") else "") + "\n"
                for r in plan["flow"]))
        if plan.get("script_map"):
            (notes / "script_map.md").write_text("# Approved script map\n\n" + "\n\n".join(
                f"## Step {step['n']}: {step['step']}\n" + "\n".join(
                    f"[{line['sentence']}] {line['text']}" for passage in step.get("passages", []) for line in passage.get("lines", []))
                for step in plan["script_map"]) + "\n")
        credit_lines = ([clip["caption"].strip()] if (clip.get("caption") or "").strip() else []) + (
            ["Pictures: " + "; ".join(plan["gfx_credits"])] if plan.get("gfx_credits") else []) + (            # CC BY needs the credit in the video's description
            ["Stock footage (Pixabay / Pexels: no credit required, given anyway):\n" + "\n".join(plan["stock_credits"])] if plan.get("stock_credits") else [])
        try:                                                                 # a song that came in through Music › Find my downloads: where it is licensed from
            from .musicfind import credit_line_for
            lic = credit_line_for((plan.get("music") or {}).get("file"))
            if lic:
                credit_lines.append("Music (from your licensed library): " + lic)
        except Exception:  # noqa: BLE001 - a credit line is a bonus
            pass
        if credit_lines:
            (notes / "credits.txt").write_text("\n\n".join(credit_lines) + "\n")
        out_words, dur = [], 0.0
        try:                                                                 # YouTube chapters and a description, from the finished video's words
            iv = [tuple(x) for x in (json.loads((d / "plan.json").read_text()).get("intervals") or [])] if (d / "plan.json").exists() else []
            out_words = remap_words(words, iv, 1.0) if iv else []
            dur = float(clip.get("duration") or 0) or sum(b - a for a, b in iv)
        except Exception as e:  # noqa: BLE001
            log.warning(f"chapters input failed: {e}")
        writing = bool(out_words and dur >= NOTES_MIN_S)
        self.projects.update(pid, notes_info={"chapters": "", "qa": qa_counts, "issues": qa_rows, "notes_state": "writing" if writing else "done",
                                              "report": str(notes / "Quality check.md"), "folder": str(notes)})
        if writing:                                                          # the AI writing runs AFTER the video is marked done (it can take minutes)
            self.start_notes(pid, out_words, dur, notes, credit_lines, (proj.get("notes") or "")[:400])

    def _log_note(self, folder: Path) -> list[dict]:
        """Which clips were shot in a log profile (converted in the MP4; Final Cut has them as shot)."""
        try:
            found: dict[str, list[str]] = {}
            for f in State(folder).get("files.json") or []:
                if f.get("photo") or not f.get("path") or not LC.vf(f["path"], self.cfg):
                    continue
                found.setdefault(str(LC.detect(f["path"]).get("profile") or "log"), []).append(Path(f["path"]).name)
            return [{"level": "info", "what": f"{len(v)} clip{'s' if len(v) != 1 else ''} shot in {k.replace('slog', 'S-Log').replace('clog', 'C-Log').replace('vlog', 'V-Log').replace('dlog', 'D-Log')} "
                                              f"({', '.join(v[:4])}{'…' if len(v) > 4 else ''}): converted to normal colour in the MP4",
                     "do": "In Final Cut these clips are as shot: add a LUT or colour-grade them there."} for k, v in found.items()]
        except Exception:  # noqa: BLE001
            return []

    # -- plain-words changes -----------------------------------------------------------------
    def revise(self, pid: str, text: str = "", revert: bool = False) -> None:
        proj, folder, st, ai, engine = self._setup(pid)
        job = proj["job"]
        P = lambda status, stage, frac: self._prog(engine, job, status, stage, frac)       # noqa: E731
        files, group, all_words, words = self.saved_inputs(folder, st, ai, self.store.get(job)["options"].get("transcriber", "parakeet"))
        scenes = self.describe(files, words, ai, st, lambda f, m: None)
        sents = seg(words)
        plan = st.get("plan.json")
        if not plan:
            raise RuntimeError("make the video first")
        if plan.get("keep_cut") and not revert:                            # a kept video: a change is about its B-roll, placed again with it
            if not st.get("plan_v1.json"):
                st.put("plan_v1.json", {k: v for k, v in plan.items() if k not in ("pending_render", "pending_note")})
            plan["broll_notes"] = [*plan.get("broll_notes", []), " ".join(str(text or "").split())[:300]]
            st.put("render_cache.json", {})                                 # (the placement is chosen afresh, with the note)
            plan["pending_render"], plan["pending_note"] = True, f"B-roll placed again: {text[:80]}"
            st.put("plan.json", plan)
            self.render(pid, plan, files, words, scenes, ai, engine, st, folder, note=plan["pending_note"], all_words=all_words)
            return
        if revert:
            first = st.get("plan_v1.json")
            if not first:
                raise RuntimeError("there is no first cut saved to go back to")
            plan = {**first, "version": plan.get("version", 1), "history": plan.get("history", [])}
            note = "back to the first cut"
        else:
            P("finding", "Understanding your change", 0.1)
            if not st.get("plan_v1.json"):
                st.put("plan_v1.json", {k: v for k, v in plan.items() if k not in ("pending_render", "pending_note")})
            ops = parse_request(ai, text, plan, self.cur_len(plan, sents, words))
            plan, note = self.apply(ops, plan, sents, words, ai, folder, st, text, P)
            try:                                                            # what you ask to change can teach the editor a general rule
                self.projects.learn(text, plan.get("title", ""), fallback=False, ai=ai, change=True)
            except Exception:  # noqa: BLE001 - learning is a bonus, never a reason to fail
                pass
        outline = FL.steps(self.projects.get(pid).get("notes") or "")
        if outline:                                                         # the script check is redone for THIS cut (the cut stays as you asked)
            P("finding", "Checking the changed cut against your script", 0.4)
            plan["flow"] = FL.recheck(ai, sents, plan["items"], outline, old=plan.get("flow"))
        plan["pending_render"], plan["pending_note"] = True, note
        before = st.get("plan.json")
        if before:
            st.snapshot("plan.json", before, "before-revision")
        st.put("plan.json", plan)                                           # saved before rendering: a failed render can be redone without asking again
        self.projects.set_notes(pid, self.projects.get(pid).get("notes", ""))
        self.render(pid, plan, files, words, scenes, ai, engine, st, folder, note=note, all_words=all_words)

    def cur_len(self, plan, sents, words) -> float:
        return sum(b - a for a, b in self.intervals_of(plan, sents, words))

    def apply(self, ops: dict, plan: dict, sents, words, ai, folder: Path, st: State, text: str, P) -> tuple[dict, str]:
        did: list[str] = []
        plan = json.loads(json.dumps(plan))
        cur = self.cur_len(plan, sents, words)
        # 1. what is said / how long it is
        target = None
        if ops.get("target_total_s"):
            target = float(ops["target_total_s"])
        elif ops.get("shorter_by_s") or ops.get("shorter_pct"):
            cut = float(ops.get("shorter_by_s") or 0) or cur * float(ops.get("shorter_pct") or 0) / 100.0
            target = max(5.0, cur - cut)
        elif ops.get("longer_by_s"):
            target = cur + float(ops["longer_by_s"])
        instr = (ops.get("content_instruction") or "").strip()
        if ops.get("move_to_front_quote"):
            instr += f" Put the part where they say \"{ops['move_to_front_quote']}\" first within its current script step; never reorder script steps."
        if target and abs(target - cur) > 1.0:
            if target < cur:
                instr += (f" Make the whole video about {target:.0f} seconds long (it is {cur:.0f} s now) by removing the weakest, "
                          "least necessary sentences. Keep the strongest parts and the hook.")
            else:
                plan = self.grow(plan, sents, words, ai, target, st, P)
                did.append(f"longer, about {target:.0f}s")
                instr = instr.strip()
        if instr.strip() and not (target and target > cur and not ops.get("content_instruction")):
            plan, d = self.edit_content(plan, sents, ai, instr.strip(), target if (target and target < cur) else None)
            if d:
                did.append(d)
        # 2. the song, and the beat
        if ops.get("change_music") or ops.get("music_request"):
            plan["music"] = {**plan.get("music", {}), "file": None, "want": ops.get("music_request", "")}
            m = plan["music"]
            t = self.choose_track(plan, folder, ai, words, plan["title"], force_new=True, want=str(ops.get("music_request") or ""), st=st)
            did.append(f"new song ({t.title})" if t else "no music found")
            if t:
                plan["music"] = {**plan["music"], "file": str(t.path), "title": t.title}
        if ops.get("beat_sync") in ("on", "off"):
            plan["beat_sync"] = ops["beat_sync"] == "on"
            did.append("cutting to the beat" if plan["beat_sync"] else "no beat sync")
        elif ops.get("change_music") and plan.get("beat_sync"):
            did.append("re-cut to the new song's beat")
        # 3. the hook
        if (ops.get("hook_text") or "").strip():
            plan["hook_text"] = " ".join(ops["hook_text"].split())[:70]
            did.append(f"hook: {plan['hook_text']}")
        elif ops.get("better_hook"):
            plan["hook_text"] = self.hook(ai, plan, sents, "", note=text) or plan.get("hook_text", "")
            did.append(f"new hook: {plan['hook_text']}")
        return plan, (ops.get("summary") or "; ".join(did) or text[:80])

    def edit_content(self, plan: dict, sents, ai, instruction: str, max_len: float | None):
        """Cuts / moves inside the current edit, in the creator's words (the same plain-words editor the short clips use)."""
        from . import edit as E
        by = {s.idx: s for s in sents}
        cur_ranges = [(it["a"], it["b"]) for it in plan["items"] if it["kind"] == "talk"]
        win = [s for a, b in cur_ranges for s in sents if a <= s.idx <= b]
        if not win or len(win) > SENTENCE_LIMIT:
            return plan, "(that edit is too big to apply by typing for a cut this long; use a shorter target length)"
        res = E.plan(ai, win, 0.0, 1e12, cur_ranges, instruction, plan["title"], max_s=max(36000.0, (max_len or 0) * 2))
        if res.problem and not res.summary:
            return plan, f"(not applied: {res.problem})"
        step_of = {i: it["step"] for it in plan["items"] if it["kind"] == "talk" and it.get("step") for i in range(it["a"], it["b"] + 1)}
        talk = []
        for a, b in res.ranges:
            # An AI edit may span a script boundary. Split it there instead of assigning the entire thought to whichever
            # step happens to have a majority, then put all step passages back in the creator's required order.
            run_a, tag = a, step_of.get(a)
            for i in range(a + 1, b + 2):
                next_tag = step_of.get(i) if i <= b else object()
                if next_tag == tag:
                    continue
                item = {"kind": "talk", "a": run_a, "b": i - 1}
                if tag:
                    item["step"] = tag
                talk.append(item)
                run_a, tag = i, next_tag
        if any(it.get("step") for it in talk):
            talk = [it for _, it in sorted(enumerate(talk), key=lambda pair: (pair[1].get("step", 10_000), pair[0]))]
        visual = [it for it in plan["items"] if it["kind"] == "visual"]
        plan["items"] = merge_items(talk, visual, [(it["a"], it["b"]) for it in plan["items"] if it["kind"] == "talk"], by)
        if res.title:
            plan["hook_text"] = res.title
        return plan, res.summary or "edited"

    def grow(self, plan, sents, words, ai, target: float, st: State, P):
        """Longer: each kept passage takes in the sentence just after it (its thought carried on), then just before it, round after round
        until the video is long enough - IN PLACE, so what is added plays next to what it belongs with. A sentence another passage already
        tells, or one that says the same as something kept, is skipped."""
        by = {s.idx: s for s in sents}
        items = [dict(it) for it in plan["items"]]
        have = {i for it in items if it["kind"] == "talk" for i in range(it["a"], it["b"] + 1)}
        need = target - sum(b - a for a, b in self.intervals_of({**plan, "items": items}, sents, words))
        for _round in range(40):
            if need <= 0:
                break
            grew = False
            for it in items:
                if it["kind"] != "talk" or need <= 0:
                    continue
                for side in ("b", "a"):
                    j = it["b"] + 1 if side == "b" else it["a"] - 1
                    if j not in by or j in have:
                        continue
                    trial = no_repeats([*items, {"kind": "talk", "a": j, "b": j, "_new": True}], sents)
                    if not any(x.get("_new") for x in trial):                 # says what is already said: not added
                        have.add(j)
                        continue
                    it[side] = j
                    have.add(j)
                    need -= by[j].end - by[j].start
                    grew = True
                    break                                                     # one sentence per passage per round: spread evenly
            if not grew:
                break
        plan["items"] = finish_sentence(items, sents)
        return plan


# ----------------------------------------------------------------------------------------- helpers
def word_limits(ref: list[dict], starts: list[float], words: list[dict], a: float, b: float) -> tuple[float, float]:
    """(earliest start, latest end) for the stretch a..b: just after the word said before its first word, just before the word said after
    its last one. `words` = the words the stretch is made of; `ref` = EVERY word (sorted, with their `starts`)."""
    own = [w for w in words if w["end"] > a and w["start"] < b]
    if not own:
        return -1e9, 1e9
    first, last = min(own, key=lambda w: w["start"]), max(own, key=lambda w: w["end"])
    i = bisect.bisect_left(starts, first["start"] - 1e-6) - 1                       # the word said just before
    j = bisect.bisect_right(starts, last["start"] + 1e-6)                           # the word said just after
    lo = min(first["start"], ref[i]["end"] + 0.03) if i >= 0 else -1e9
    hi = max(last["end"], ref[j]["start"] - 0.04) if j < len(ref) else 1e9
    return lo, hi


def refresh_timecodes(files: list[dict], group: dict | None) -> None:
    """Re-read each file's start timecode (frame-exact, see timecode_s): files.json and multicam.json saved before 2026-10-13 hold the
    old, label-based value, which put clips 10-70 s off in Final Cut for 23.976 footage."""
    cache: dict[str, tuple[float, bool]] = {}

    def tc_of(path: str, fps) -> tuple[float, bool]:
        if path not in cache:
            try:
                from .polish import _fps
                info = ffmpeg.probe(path)
                vs = ffmpeg.video_stream(info) or {}
                cache[path] = (timecode_s(info, _fps(vs) if vs else fps), timecode_df(info))
            except Exception:  # noqa: BLE001 - unreadable: leave what was saved
                cache[path] = (None, False)
        return cache[path]
    for f in files:
        if f.get("path") and not f.get("photo") and not f.get("multicam"):
            tc, df = tc_of(f["path"], f.get("fps"))
            if tc is not None:
                f["tc"], f["tc_df"] = tc, df
    for a in (group or {}).get("angles") or []:
        for c in a["files"]:
            tc, df = tc_of(c["path"], c.get("fps"))
            if tc is not None:
                c["tc"], c["tc_df"] = tc, df


def script_note(flow: list[dict] | None) -> str:
    """The script check, for the job's status line."""
    if not flow:
        return ""
    bad = [r for r in flow if r.get("missing") or r.get("not_in_footage") or (not r.get("ranges") and not r.get("rechecked"))]
    if not bad:
        return " - every point of your script is in it"
    parts = []
    for r in bad:
        if r.get("missing"):
            parts.append(f"step {r['n']} missing " + ", ".join(r["missing"][:3]))
        elif r.get("not_in_footage"):
            parts.append(f"step {r['n']} not in the footage: " + ", ".join(r["not_in_footage"][:3]))
        else:
            parts.append(f"step {r['n']} has nothing in the cut")
    return " - script check: " + "; ".join(parts)


def items_from_intervals(intervals, sents: list[M.Sentence]) -> list[dict]:
    """The story planner's kept stretches (virtual seconds) -> runs of sentences, plus stretches that have no speech."""
    kept = [s for s in sents if any(a - 0.05 <= (s.start + s.end) / 2 <= b + 0.05 for a, b in intervals)]
    talk, run = [], []
    for s in kept:
        if run and s.idx != run[-1].idx + 1:
            talk.append({"kind": "talk", "a": run[0].idx, "b": run[-1].idx, "t": run[0].start})
            run = []
        run.append(s)
    if run:
        talk.append({"kind": "talk", "a": run[0].idx, "b": run[-1].idx, "t": run[0].start})
    visual = [{"kind": "visual", "start": a, "end": b, "t": a} for a, b in intervals
              if not any(a - 0.05 <= (s.start + s.end) / 2 <= b + 0.05 for s in sents)]
    items = sorted(talk + visual, key=lambda it: it["t"])
    for it in items:
        it.pop("t", None)
    return items


def merge_items(talk: list[dict], visual: list[dict], old_order, by, keep_order: bool = True) -> list[dict]:
    """New talk ranges (in the order the editor chose) with the silent-footage stretches put back where they belong in time."""
    out = [dict(t) for t in talk]
    for v in sorted(visual, key=lambda v: v["start"]):
        pos = len(out)
        for i, t in enumerate(out):
            if t["kind"] == "talk" and t["a"] in by and by[t["a"]].start > v["start"]:
                pos = i
                break
        out.insert(pos, v)
    return out



# ----------------------------------------------------------------------------------------- multicam + pieces of the program
def group_sig(files: list[dict], audios: list[dict]) -> str:
    key = [(f["name"], f["size"], f["mtime"]) for f in files if f["audio"] and f["dur"] >= 15 and not f.get("broll")]
    key += [(a["name"], a["size"], a["mtime"]) for a in audios]
    return hashlib.sha1(json.dumps(["v5", *sorted(key)]).encode()).hexdigest()           # v2: cameras told apart by sound; v4: sample-accurate sync; v5: a camera with almost no sound is matched after raising its level


def virtualize(files: list[dict], group: dict) -> list[dict]:
    """The multicam shoot becomes ONE long 'file' at the start of the timeline (its time = the shared timeline); everything else follows."""
    gnames = set(group["files"])
    main = max(group["angles"], key=lambda a: sum(c["dur"] for c in a["files"]))["files"][0]
    pseudo = {"name": "Multicam (" + ", ".join(a["id"] for a in group["angles"]) + ")", "path": None, "rel": "", "size": 0, "mtime": 0,
              "dur": group["dur"], "start": 0.0, "end": group["dur"], "w": main["w"], "h": main["h"], "fps": main["fps"], "audio": True,
              "broll": False, "multicam": True}
    short = set(group.get("short") or [])
    out, t = [pseudo], group["dur"]
    for f in files:
        if f["name"] in gnames:
            continue
        out.append({**f, "start": round(t, 3), "end": round(t + f["dur"], 3), **({"broll": True} if f["name"] in short else {})})
        t += f["dur"]
    return out


SOUND_NOTE = ("\nSome moments are marked [poor sound] (recorded far from the mic, muffled or very quiet) or [so-so sound]. Never open the "
              "video with a [poor sound] moment, avoid them unless the line really matters, and prefer a well-recorded moment that says the same.")


def sound_source(files: list[dict], group: dict | None, t: float) -> tuple[str, float] | None:
    """Which recording is heard at program second t, and where in it: (path, seconds into it)."""
    f = next((f for f in files if f["start"] <= t < f["end"] and not f.get("photo")), None)
    if f is None:
        return None
    loc = t - f["start"]
    if f.get("multicam"):
        s = MC.sound_at(group, loc) if group else None
        return (s["path"], s["local"]) if s else None
    if f.get("clean"):
        return f["clean"]["audio"], loc + float(f["clean"]["offset"])
    if f.get("audio") and f.get("path"):
        return f["path"], loc
    return None


FILLER_WORD = re.compile(r"^(u+m+|u+h+|e+r+m*|a+h+|h+m+|m+h*m+)[,.!?]*$", re.I)


def merge_tiny(intervals, all_words: list[dict] | None = None, min_s: float = 1.2, max_gap: float = 1.0) -> list[tuple[float, float]]:
    """A kept piece shorter than min_s seconds (a word or two left between two cuts) joins the piece before or after it when only a short
    pause or a filler lies between them - a 0.7 s piece is a flash, not a shot. Never across anyone else's words."""
    iv = [list(x) for x in intervals]

    def quiet(x: float, y: float) -> bool:
        return y - x <= max_gap and not any(x + 0.02 < float(w["start"]) < y - 0.02 and not FILLER_WORD.match(str(w.get("text", "")).strip())
                                            for w in (all_words or []))
    k = 0
    while k < len(iv):
        a, b = iv[k]
        if b - a < min_s and len(iv) > 1:
            if k > 0 and 0 <= a - iv[k - 1][1] and quiet(iv[k - 1][1], a):
                iv[k - 1][1] = b
                iv.pop(k)
                continue
            if k + 1 < len(iv) and 0 <= iv[k + 1][0] - b and quiet(b, iv[k + 1][0]):
                iv[k + 1][0] = a
                iv.pop(k)
                continue
        k += 1
    return [(a, b) for a, b in iv]


def opening_without_bad_sound(intervals, grades: list[dict | None], opening: float = 20.0) -> tuple[list, list[int]]:
    """The first `opening` seconds of the video never play badly recorded sound: such stretches there are taken out (when most of the
    video sounds fine; a video that is all one bad recording keeps it). Returns (intervals, the indexes taken out)."""
    total = sum(b - a for a, b in intervals) or 1.0
    bad_s = sum(b - a for (a, b), g in zip(intervals, grades) if g and g["grade"] == "bad")
    if bad_s / total >= 0.5:
        return list(intervals), []
    out, dropped, t = [], [], 0.0
    for k, ((a, b), g) in enumerate(zip(intervals, grades)):
        if t < opening and g and g["grade"] == "bad" and len(intervals) - len(dropped) > 1:
            dropped.append(k)
            continue
        out.append((a, b))
        t += b - a
    return out, dropped


def level_gains(levels: list[float], weights: list[float], most_up: float = 12.0, most_down: float = 10.0) -> list[float]:
    """dB for each stretch so its speech comes out at the (time-weighted) median level of the video: no stretch 15 dB louder than the
    rest. Small differences (under 1.5 dB) are left alone."""
    pairs = sorted((lv, w) for lv, w in zip(levels, weights) if lv > -90)
    if not pairs:
        return [0.0] * len(levels)
    half, acc, target = sum(w for _, w in pairs) / 2, 0.0, pairs[-1][0]
    for lv, w in pairs:
        acc += w
        if acc >= half:
            target = lv
            break
    out = []
    for lv in levels:
        g = 0.0 if lv <= -90 else max(-most_down, min(most_up, target - lv))
        out.append(round(g, 1) if abs(g) >= 1.5 else 0.0)
    return out


def no_picture_spans(files: list[dict], group: dict | None, intervals) -> list[tuple[float, float]]:
    """Kept stretches (program time) of a multicam shoot that no camera filmed - only a recorder heard them. Shown as they are, the voice
    plays over whichever camera happens to be there (a workshop shoot, 2026-11-05: a man on screen, someone else's voice, -ss -938)."""
    if not group:
        return []
    cov: list[list[float]] = []
    for c0, c1 in sorted((float(c["t0"]), float(c["t0"]) + float(c["dur"])) for a in group["angles"] for c in a["files"]):
        if cov and c0 <= cov[-1][1] + 0.02:
            cov[-1][1] = max(cov[-1][1], c1)
        else:
            cov.append([c0, c1])
    out: list[tuple[float, float]] = []
    for f in files:
        if not f.get("multicam"):
            continue
        for a, b in intervals:
            x0, x1 = max(a, f["start"]), min(b, f["end"])
            if x1 - x0 <= 0.02:
                continue
            t, m1 = x0 - f["start"], x1 - f["start"]
            for c0, c1 in cov:
                if c1 <= t:
                    continue
                if c0 >= m1:
                    break
                if c0 > t:
                    out.append((t + f["start"], min(c0, m1) + f["start"]))
                t = max(t, c1)
                if t >= m1:
                    break
            if t < m1:
                out.append((t + f["start"], m1 + f["start"]))
    return [(round(x, 3), round(y, 3)) for x, y in out if y - x > 0.05]


def subtract_spans(intervals, spans, min_len: float = 0.5) -> list[tuple[float, float]]:
    """The intervals with the spans cut out (pieces shorter than min_len dropped)."""
    out = []
    for a, b in intervals:
        pieces = [(a, b)]
        for x, y in spans:
            nxt = []
            for p0, p1 in pieces:
                if y <= p0 or x >= p1:
                    nxt.append((p0, p1))
                    continue
                if x > p0:
                    nxt.append((p0, x))
                if y < p1:
                    nxt.append((y, p1))
            pieces = nxt
        out += [(p0, p1) for p0, p1 in pieces if p1 - p0 >= min_len]
    return out


def out_time(intervals, t: float) -> float | None:
    """A program (source-timeline) time -> its time in the finished cut (None when it isn't kept)."""
    acc = 0.0
    for a, b in intervals:
        if a - 1e-6 <= t <= b + 1e-6:
            return acc + min(max(t, a), b) - a
        acc += b - a
    return None


def _covered(cutaways: list[dict], s0: float, s1: float) -> bool:
    """Is every moment of s0..s1 under some cutaway (back to back counts)?"""
    t = s0
    for c in sorted(cutaways, key=lambda c: c["at"]):
        if c["at"] <= t + 0.02 and c["at"] + c["length"] > t:
            t = c["at"] + c["length"]
        if t >= s1 - 0.02:
            return True
    return t >= s1 - 0.02


def cover_spans(cutaways: list[dict], spans_out: list[tuple[float, float]], pool: list[dict], exclude: set[int] = frozenset(),
                choose=None, show: float = 3.5, why: str = "nobody was filmed while this was said") -> tuple[list[dict], list[tuple[float, float]]]:
    """B-roll / photos over the WHOLE of each span of the cut (finished-video seconds): shots of up to `show` seconds back to back, each a
    little past the span's edges. Cutaways already overlapping a span are trimmed out of it. Returns (cutaways, spans it could not cover)."""
    cutaways = [dict(c) for c in cutaways]
    used = {c["scene"] for c in cutaways} | set(exclude)
    free = [i for i in sorted(range(len(pool)), key=lambda i: -pool[i]["quality"]) if i not in used]
    missed = []
    for s0, s1 in spans_out:
        a0, a1 = max(0.0, s0 - 0.15), s1 + 0.15
        if _covered(cutaways, s0, s1):
            continue
        kept = []
        for c in cutaways:                                               # make room: what overlaps the span ends before it (or starts after)
            c0, c1 = c["at"], c["at"] + c["length"]
            if c1 <= a0 or c0 >= a1:
                kept.append(c)
            elif c0 < a0 and a0 - c0 >= 1.2:
                kept.append({**c, "length": round(a0 - c0, 2)})
            elif c1 > a1 and c1 - a1 >= 1.2:
                kept.append({**c, "at": round(a1, 2), "length": round(c1 - a1, 2), "src": round(c["src"] + (a1 - c0), 2)})
        cutaways = kept
        t, new = a0, []
        while t < a1 - 0.05:
            if not free:                                                 # out of shots: one placed for taste gives way (this one is needed)
                victim = next((c for c in reversed(cutaways) if not c.get("must") and not str(c.get("why", "")).startswith(PROTECTED)
                               and (c["at"] + c["length"] <= a0 or c["at"] >= a1)), None)
                if victim is None:
                    break
                cutaways.remove(victim)
                free.append(victim["scene"])
            pick = choose(t, free) if choose else None
            if pick is None or pick not in free:
                pick = free[0]
            free.remove(pick)
            sc = pool[pick]
            room = (a1 - t) if sc.get("photo") else min(a1 - t, sc["end"] - sc["start"])
            ln = min(show, room) if a1 - t - show > 1.2 else room           # (no stub of a shot at the end)
            if ln < 0.6:
                continue
            new.append({"at": round(t, 2), "length": round(ln, 2), "src": round(sc["start"], 2), "scene": pick, "what": sc["what"],
                        "why": why, "file": sc.get("file"), "photo": sc.get("photo"), "must": True})
            t += ln
        if t < a1 - 0.05:
            missed.append((s0, s1))
            continue
        cutaways += new
    cutaways.sort(key=lambda c: c["at"])
    return cutaways, missed


PACE_BROLL = {"fast": 0.6, "tight": 0.85, "natural": 1.0, "slow": 1.6}
JUMP_SHARE = {"fast": 0.75, "tight": 0.65, "natural": 0.6, "slow": 0.45}      # at most this share of the video under B-roll from jump covers      # pace -> how often B-roll cuts in (x the kind's cut_per_s)
PROTECTED = ("cover", "never the same picture", "nobody was filmed", "trimmed by you")          # cutaways that are there for a reason, not taste


def _room(c: dict, pool: list[dict]) -> tuple[float, float]:
    """How much a cutaway's shot could grow: (seconds before its start, seconds after its end) still inside the shot. A photo: any."""
    if c.get("photo"):
        return 1e9, 1e9
    if str(c.get("why", "")).startswith("trimmed by you"):                  # (trimmed by hand: the automatic gap closing must not undo it)
        return 0.0, 0.0
    sc = pool[c["scene"]] if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(pool) else None
    if not sc:
        return 0.0, 0.0
    return max(0.0, float(c["src"]) - float(sc["start"])), max(0.0, float(sc["end"]) - (float(c["src"]) + float(c["length"])))


def thin_cutaways(cutaways: list[dict], pace: str = "natural", min_between: float | None = None) -> list[dict]:
    """Taste, not rules: B-roll that would show the speaker for only a moment between two shots (1.5 s to `min_between`) is dropped, so
    the video breathes (a promo got a new shot every 3-4 s through its whole first minute). Fast-paced keeps them; protected cutaways
    (jump covers, the twelve-second rule, sound nobody filmed) always stay. Gaps under 1.5 s are closed by no_flash_gaps instead."""
    mb = min_between if min_between is not None else (1.5 if pace == "fast" else 3.0)
    out: list[dict] = []
    for c in sorted(cutaways, key=lambda c: c["at"]):
        if out:
            gap = c["at"] - (out[-1]["at"] + out[-1]["length"])
            if 1.5 <= gap < mb and not c.get("must") and not str(c.get("why", "")).startswith(PROTECTED):
                continue
        out.append(dict(c))
    return out


def no_flash_gaps(cutaways: list[dict], pool: list[dict], out_len: float, min_gap: float = 1.5, min_show: float = 1.2) -> list[dict]:
    """Never a flash of the speaker between two B-roll shots (a workshop shoot: photos at 60, 63, 66 s, each 2.5 s, left 0.5 s of the man
    in green each time): a gap under `min_gap` is closed - the first shot runs on, or the next starts earlier from earlier in its shot, or
    both share it. A gap that can't be closed and is under `min_show` loses the later (unprotected) shot. The same at the two ends."""
    cs = sorted((dict(c) for c in cutaways), key=lambda c: c["at"])
    k = 0
    while k + 1 < len(cs):
        c, n = cs[k], cs[k + 1]
        gap = round(n["at"] - (c["at"] + c["length"]), 3)
        if gap <= 0.0 or gap >= min_gap:
            if gap < 0:                                                     # overlapping: the first ends where the next starts...
                if n["at"] - c["at"] < min_show:                                # ...unless that leaves a flash of it: keep one of the two
                    drop = c if not (c.get("must") or str(c.get("why", "")).startswith(PROTECTED)) else n
                    cs.remove(drop)
                    continue
                c["length"] = round(n["at"] - c["at"], 2)
            k += 1
            continue
        after_c = _room(c, pool)[1]
        before_n = _room(n, pool)[0]
        give_c = min(gap, after_c)
        give_n = min(gap - give_c, before_n)
        if give_c + give_n >= gap - 1e-3:
            c["length"] = round(c["length"] + give_c, 2)
            if give_n > 0:
                n["at"], n["src"], n["length"] = round(n["at"] - give_n, 2), round(n["src"] - give_n, 2), round(n["length"] + give_n, 2)
            k += 1
            continue
        if gap < min_show:
            drop = n if not (n.get("must") or str(n.get("why", "")).startswith(PROTECTED)) else (
                c if not (c.get("must") or str(c.get("why", "")).startswith(PROTECTED)) else None)
            if drop is not None:
                cs.remove(drop)
                continue
        k += 1
    if cs:                                                                  # the very end: no last moment of the speaker after a shot
        last = cs[-1]
        tail = out_len - (last["at"] + last["length"])
        if 0 < tail < min_show and _room(last, pool)[1] >= tail:
            last["length"] = round(last["length"] + tail, 2)
        first = cs[0]
        if 0 < first["at"] < 0.6 and _room(first, pool)[0] >= first["at"]:  # ...nor a sliver of him before the first
            first["src"], first["length"], first["at"] = round(first["src"] - first["at"], 2), round(first["length"] + first["at"], 2), 0.0
    return cs


def jump_cuts(intervals, files: list[dict], cuts: list[dict] | None = None, min_jump: float = 0.25) -> list[tuple[float, float]]:
    """Where the cut jumps within one shot - the same file (or the same camera angle of a shoot) continuing from a later moment: the
    picture twitches while the person stays put. -> [(program second where the later piece starts, seconds skipped)], biggest first."""
    def file_at(t: float):
        return next((f for f in files if f["start"] <= t < f["end"] and not f.get("photo")), None)

    def angle(t: float):
        return next((c["angle"] for c in (cuts or []) if c["start"] - 1e-6 <= t < c["end"]), None)
    out = []
    for (a1, b1), (a2, b2) in zip(intervals, intervals[1:]):
        if abs(a2 - b1) < min_jump:
            continue
        f1, f2 = file_at(b1 - 1e-3), file_at(a2 + 1e-3)
        if not f1 or not f2 or f1["name"] != f2["name"] or f1.get("broll"):
            continue
        if f1.get("multicam") and angle(b1 - f1["start"] - 1e-3) != angle(a2 - f1["start"] + 1e-3):
            continue                                                       # another camera takes over at the cut: no jump to see
        out.append((round(a2, 3), round(abs(a2 - b1), 3)))
    return sorted(out, key=lambda x: -x[1])


def picture_durations(files: list[dict], group: dict | None, outro: dict | None = None) -> dict[str, float]:
    """How long each picture file is (to check that no piece reads before its start or past its end)."""
    d = {f["path"]: float(f["dur"]) for f in files if f.get("path") and f.get("dur")}
    for a in (group or {}).get("angles", []):
        for c in a["files"]:
            d[c["path"]] = float(c["dur"])
    if outro and outro.get("path") and outro.get("file_dur"):
        d[outro["path"]] = float(outro["file_dur"])
    return d


def _shots(parts: list[dict]) -> list[list[int]]:
    """Runs of pieces that are one continuous shot on screen (the same file, carrying straight on)."""
    out: list[list[int]] = []
    for i, p in enumerate(parts):
        if out:
            q = parts[out[-1][-1]]
            if p["v"].get("path") and p["v"].get("path") == q["v"].get("path") and abs(float(p["v"]["ss"]) - (float(q["v"]["ss"]) + q["d"])) < 0.06:
                out[-1].append(i)
                continue
        out.append([i])
    return out


def _same_moment(parts: list[dict], i0: int, i1: int) -> bool:
    """Do pieces i0..i1 carry ONE recording straight on (no jump in what is heard)? Then a speaker's picture may run on over them."""
    for j in range(i0 + 1, i1 + 1):
        a, b = parts[j - 1].get("a"), parts[j].get("a")
        if not a or not b or a["path"] != b["path"] or abs(float(b["ss"]) - (float(a["ss"]) + parts[j - 1]["d"])) > 0.06:
            return False
    return True


def fix_flashes(parts: list[dict], durs: dict[str, float], min_s: float = 1.0) -> tuple[list[dict], list[float]]:
    """The last look at the cut as pieces: no shot on screen shorter than `min_s` (a half-second flash of the speaker between two B-roll
    shots, a sliver of another camera). The shot before runs on over it, or the one after starts earlier - B-roll freely, a speaker's own
    picture only where the sound carries straight on (so the lips still match). Returns (parts, where flashes are left, in seconds)."""
    for _ in range(len(parts)):
        sh = _shots(parts)
        if len(sh) < 2:
            break
        changed = False
        for k, idx in enumerate(sh):
            d = sum(parts[i]["d"] for i in idx)
            if d >= min_s:
                continue
            for side in ("prev", "next"):
                if side == "prev" and k > 0:
                    src = parts[sh[k - 1][-1]]
                    ok = src["v"].get("cut") or _same_moment(parts, sh[k - 1][-1], idx[-1])
                    off = float(src["v"]["ss"]) + src["d"]
                    fits = src["v"].get("path") and off + d <= durs.get(src["v"]["path"], 1e9) - 0.05
                elif side == "next" and k + 1 < len(sh):
                    src = parts[sh[k + 1][0]]
                    ok = src["v"].get("cut") or _same_moment(parts, idx[0], sh[k + 1][0])
                    off = float(src["v"]["ss"]) - d
                    fits = src["v"].get("path") and off >= 0.0
                else:
                    continue
                if ok and fits:
                    for i in idx:
                        if not parts[i]["v"].get("cut") and not parts[i].get("v0"):
                            parts[i]["v0"] = parts[i]["v"]
                        parts[i]["v"] = {**src["v"], "ss": round(off, 6)}
                        off += parts[i]["d"]
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    left, t = [], 0.0
    sh = _shots(parts)
    starts = [sum(p["d"] for p in parts[:idx[0]]) for idx in sh]
    for idx, t in zip(sh, starts):
        if len(sh) > 1 and sum(parts[i]["d"] for i in idx) < min_s:
            left.append(round(t, 2))
    return parts, left


def sound_runs(parts: list[dict]) -> list[list[int]]:
    """Runs of pieces that play one recording straight on (what is levelled as one stretch)."""
    out: list[list[int]] = []
    for i in range(len(parts)):
        if out and _same_moment(parts, out[-1][-1], i):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def bad_pictures(parts: list[dict], durs: dict[str, float]) -> list[str]:
    """The last look before anything is cut: a piece whose picture would come from before the start or after the end of its file (a
    camera that wasn't rolling) is refused."""
    bad = []
    for p in parts:
        v = p.get("v") or {}
        if v.get("nopic"):
            bad.append(f"{p['d']:.2f}s of sound that no camera filmed, with nothing over it")
            continue
        if not v.get("path"):
            continue
        ss, d, dur = float(v.get("ss") or 0.0), float(p["d"]), durs.get(v["path"])
        if ss < -0.05 or (dur is not None and ss + d > dur + 0.1):
            bad.append(f"{Path(v['path']).name} at {ss:.2f}s for {d:.2f}s (the file is {dur if dur is not None else '?'} s)")
    return bad


def creep(base: float, elapsed: float, tau: float = 240.0, most: float = 0.04) -> float:
    """Where the bar shows `elapsed` seconds into a step that started at `base`: a few percent on, approached but never reached, so a long
    step visibly moves and the next real step's value (usually higher) is not overtaken by much."""
    import math
    span = min(most, max(0.0, 0.96 - base))
    return base + span * (1.0 - math.exp(-max(0.0, elapsed) / max(1.0, tau)))


def _angle_of_clip(group: dict, path: str) -> str | None:
    return next((a["id"] for a in group["angles"] if any(c["path"] == path for c in a["files"])), None)


def mc_parts(group: dict, cuts: list[dict], t0: float, t1: float) -> list[dict]:
    """Pieces of the program inside the multicam shoot between master seconds t0 and t1: the director's angle for the picture, the
    best sound (recorder first) for the audio. A piece never crosses a cut, or the end of any file it uses."""
    out, t = [], t0
    default = max(group["angles"], key=lambda a: sum(c["dur"] for c in a["files"]))["id"]
    while t < t1 - 0.02:
        seg = next((c for c in cuts if c["start"] - 1e-6 <= t < c["end"] - 1e-6), None)
        order = ([seg["angle"]] if seg else []) + [default] + [a["id"] for a in group["angles"]]
        pic = next(((aid, MC.cover(MC.angle_files(group, aid), t)) for aid in order if MC.cover(MC.angle_files(group, aid), t)), None)
        snd = MC.sound_at(group, t)
        if pic is None and snd is None:
            starts = [c["t0"] for a in group["angles"] for c in a["files"] if c["t0"] > t] + [r["t0"] for r in group["recorders"] if r["t0"] > t]
            t = min([t1] + starts)                                         # nothing was recording here: skip ahead
            continue
        ends = [t1]
        if seg:
            ends.append(seg["end"])
        if pic:
            ends.append(pic[1]["t0"] + pic[1]["dur"])
        if snd:
            ends.append(snd["clip"]["t0"] + snd["clip"]["dur"])
        e = min(ends)
        if e - t < 0.02:
            e = min(t1, t + 0.04)
        aud = "Recorder" if snd and snd["clip"] in group["recorders"] else (_angle_of_clip(group, snd["path"]) if snd else None)
        if pic is None:                                                      # sound but no picture: no camera may stand in for it (its frame
            out.append({"d": e - t, "v": {"path": None, "ss": 0.0, "fill": False, "nopic": True},     # would be another moment, another
                        "a": ({"path": snd["path"], "ss": snd["local"], "pre": 0.0} if snd else None),    # person): B-roll covers it (cover_spans)
                        "item": {"kind": "mc", "t": t, "video": None, "audio": aud}})
            t = e
            continue
        aid, c = pic
        out.append({"d": e - t, "v": {"path": c["path"], "ss": t - c["t0"], "fill": False},
                    "a": ({"path": snd["path"], "ss": snd["local"], "pre": 0.0} if snd else None),
                    "item": {"kind": "mc", "t": t, "video": aid, "audio": aud or aid}})
        t = e
    return out


LAYOUT_STYLES = ("story", "business", "youtube")   # kinds of video where the speaker's face carries the message (never an ad)
PIP_MIN_S = 6.0                              # a B-roll run this long over speech: now and then the speaker comes back in a window
SPLIT_MIN_S = 3.0                            # vertical videos: speaker on top, B-roll below
LAYOUT_EVERY = 3                             # at most one in three of the shots that qualify (sparingly)
from .fcpxml import PIP_MARGIN, PIP_SCALE    # (the same window size and place as the Final Cut project's)


def choose_layouts(cutaways: list[dict], W: int, H: int, style: str | None, on: bool = True) -> list[dict]:
    """Picture-in-picture and split screen, sparingly (the research list, 2026-11-06): in a horizontal video a long B-roll run over
    speech (PIP_MIN_S+) sometimes keeps the speaker in a small window top right (away from the captions at the bottom and the lower
    thirds bottom left); in a vertical one a B-roll shot sometimes shares the frame, the speaker on top and the B-roll below. Only for
    kinds of video where the face matters, never for a shot that covers a moment nobody filmed (there is no speaker to show), and only
    every LAYOUT_EVERY-th qualifying shot, starting with the first. Photos qualify too (photos.bake already frames a photo that doesn't
    fit). Sets / clears c["layout"] in place; returns the list."""
    n = 0
    for c in sorted(cutaways, key=lambda c: c.get("at", 0.0)):
        c.pop("layout", None)
        if not on or (style or "story") not in LAYOUT_STYLES:
            continue
        if c.get("must") or str(c.get("why", "")).startswith(PROTECTED):
            continue
        vertical, length = H > W, float(c.get("length") or 0.0)
        if length < (SPLIT_MIN_S if vertical else PIP_MIN_S):
            continue
        if n % LAYOUT_EVERY == 0:
            c["layout"] = "split" if vertical else "pip"
        n += 1
    return cutaways


def layout_masks(d: Path, W: int, H: int) -> dict:
    """The picture-in-picture window's pieces, drawn once with Pillow: a rounded mask (grey, white = the speaker) and a soft shadow with
    a thin light border to sit it on (RGBA, bigger by `pad` on each side)."""
    from PIL import Image, ImageDraw, ImageFilter
    pw, ph = int(W * PIP_SCALE) // 2 * 2, int(H * PIP_SCALE) // 2 * 2
    r = max(6, int(ph * 0.07))
    pad = max(8, int(ph * 0.08))
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=r, fill=255)
    back = Image.new("RGBA", (pw + 2 * pad, ph + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("L", back.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((pad, pad + pad // 3, pad + pw, pad + ph + pad // 3), radius=r, fill=150)
    back.putalpha(sh.filter(ImageFilter.GaussianBlur(pad / 2.2)))
    rim = max(2, int(ph * 0.008))
    ImageDraw.Draw(back).rounded_rectangle((pad - rim, pad - rim, pad + pw - 1 + rim, pad + ph - 1 + rim), radius=r + rim,
                                           fill=(245, 245, 245, 235))
    mp, bp = d / f"pip_mask_{pw}x{ph}.png", d / f"pip_back_{pw}x{ph}.png"
    mask.save(mp)
    back.save(bp)
    m = int(H * PIP_MARGIN)
    return {"mask": mp, "back": bp, "pw": pw, "ph": ph, "pad": pad, "x": W - m - pw, "y": m}


FACE_Y = 0.35                                # no face found: a talking head's face is about a third of the way down


def face_y(path: str, ss: float, d: float, models_dir: Path) -> float:
    """Where the speaker's face is, down the frame (0..1), so a split screen's half frame keeps the head (a centre crop of a talking head
    cut it off at the top). The biggest face in a few frames; FACE_Y when there isn't one."""
    try:
        from . import reframe as R
        samples, _n = R.detect_faces(Path(path), max(0.0, float(ss)), max(0.5, min(3.0, float(d))), models_dir, rate=1.0, width=480)
        ys = [max(fs, key=lambda f: f.w).cy for fs in samples if fs]
        if ys:
            return float(sorted(ys)[len(ys) // 2])
    except Exception as e:  # noqa: BLE001 - the usual place is a fine guess
        log.warning(f"no face position for a split screen ({e})")
    return FACE_Y


def layout_graph(layout: str, W: int, H: int, rs: str, lut_b: str, lut_s: str, n: int, masks: dict | None, fy: float = FACE_Y) -> str:
    """The filter graph of a layout piece. Inputs: 0 = the B-roll, 1 = the sound, 2 = the speaker, 3 = mask, 4 = shadow (pip)."""
    end = f"tpad=stop_mode=clone:stop_duration=2,trim=end_frame={n},setpts=PTS-STARTPTS"
    if layout == "pip" and masks:
        pw, ph, pad = masks["pw"], masks["ph"], masks["pad"]
        return (f"[0:v:0]{lut_b}fps={rs},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,format=rgba,{end}[bg];"
                f"[2:v:0]{lut_s}fps={rs},scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph},setsar=1,format=rgba,{end}[sp];"
                f"[3:v]format=gray,scale={pw}:{ph}[m];[sp][m]alphamerge[spm];[4:v]format=rgba[sh];"
                f"[bg][sh]overlay={masks['x'] - pad}:{masks['y'] - pad}:shortest=1[b1];"
                f"[b1][spm]overlay={masks['x']}:{masks['y']}:shortest=1,format=yuv420p,{end}[v]")
    if layout == "split":
        hh = H // 2 // 2 * 2
        crop_y = f"max(0\\,min(ih-{hh}\\,ih*{fy:.3f}-{hh}*0.42))"           # the face about 40% down the half frame
        return (f"[2:v:0]{lut_s}fps={rs},scale={W}:{hh}:force_original_aspect_ratio=increase,crop={W}:{hh}:(iw-{W})/2:{crop_y},setsar=1,{end}[top];"
                f"[0:v:0]{lut_b}fps={rs},scale={W}:{H - hh}:force_original_aspect_ratio=increase,crop={W}:{H - hh},setsar=1,{end}[bot];"
                f"[top][bot]vstack=inputs=2,format=yuv420p,{end}[v]")
    raise ValueError(f"no such layout: {layout}")


def build_parts(files: list[dict], group: dict | None, cuts: list[dict], intervals, cutaways: list[dict], outro: dict | None = None) -> list[dict]:
    """The whole program as pieces, in order, each with its picture source and sound source (and what Final Cut needs to know)."""
    def file_at(t: float) -> dict:
        play = lambda f: not f.get("photo") or f.get("mphoto")             # noqa: E731 (a montage's photos are shots of their own)
        return next((f for f in files if f["start"] <= t < f["end"] + 1e-6 and play(f)), next(f for f in reversed(files) if play(f)))
    parts: list[dict] = []
    for a, b in intervals:
        t = a
        while t < b - 0.02:
            f = file_at(t + 1e-4)
            e = min(b, f["end"])
            if e <= t:                                                          # a kept stretch past the end of the footage: nothing more to cut
                break
            if e - t > 0.02:
                if f.get("multicam") and group:
                    parts += mc_parts(group, cuts, t - f["start"], e - f["start"])
                else:
                    loc = t - f["start"]
                    clean = f.get("clean")
                    if clean:
                        aud = {"path": clean["audio"], "ss": loc + float(clean["offset"]), "pre": 0.0}
                    elif f["audio"]:
                        aud = {"path": f["path"], "ss": loc, "pre": 0.0}
                    else:
                        aud = None
                    item = {"kind": "clip", "path": f["path"], "start": loc, "w": f["w"], "h": f["h"], "fps": f["fps"],
                            "audio": bool(f["audio"]), "file_dur": f["dur"], "tc": f.get("tc", 0.0), "tc_df": f.get("tc_df", False)}
                    if clean:                                                       # Final Cut hears the same recording the video uses
                        item["aud"] = {"path": clean["audio"], "start": aud["ss"], "tc": float(clean.get("tc") or 0.0),
                                       "dur": clean.get("dur"), "ch": clean.get("ch")}
                    parts.append({"d": e - t, "v": {"path": f["path"], "ss": loc, "fill": False}, "a": aud, "item": item})
            t = e
    pos = 0.0
    for p in parts:
        p["out"] = pos
        pos += p["d"]
    total = pos
    # B-roll cutaways replace the PICTURE for a stretch of the timeline; the sound is left alone
    for c in sorted(cutaways, key=lambda c: c["at"]):
        c0, c1 = c["at"], min(total, c["at"] + c["length"])
        bf = next((f for f in files if not f.get("multicam") and f["start"] - 1e-3 <= c["src"] < f["end"]), None)
        if bf is None or c1 - c0 < 0.3:
            continue
        nxt = []
        for p in parts:
            p0, p1 = p["out"], p["out"] + p["d"]
            if p1 <= c0 or p0 >= c1 or p["v"].get("cut"):
                nxt.append(p)
                continue
            if p0 < c0:
                nxt.append(_slice(p, 0.0, c0 - p0))
            m0, m1 = max(p0, c0), min(p1, c1)
            mid = _slice(p, m0 - p0, m1 - m0)
            mid["v0"] = mid["v"]                                            # the speaker's own picture, in case the B-roll can't be used
            ss = c["src"] - bf["start"] + (m0 - c0)
            if ss + mid["d"] > bf["dur"] - 0.05:                            # the B-roll ends before this piece does: keep the speaker
                nxt.append(mid)
                if p1 > c1:
                    nxt.append(_slice(p, c1 - p0, p1 - c1))
                continue
            mid["v"] = {"path": bf["path"], "ss": ss, "fill": True, "cut": True, **({"layout": c["layout"]} if c.get("layout") else {})}
            nxt.append(mid)
            if p1 > c1:
                nxt.append(_slice(p, c1 - p0, p1 - c1))
        parts = nxt
    parts.sort(key=lambda p: p["out"])
    if outro:                                                               # the closing shot: picture only, the music carries it
        end = parts[-1]["out"] + parts[-1]["d"] if parts else 0.0
        parts.append({"d": outro["d"], "out": end, "v": {"path": outro["path"], "ss": outro["ss"], "fill": True, "cut": True}, "a": None,
                      "item": {"kind": "clip", "path": outro["path"], "start": outro["ss"], "w": outro["w"], "h": outro["h"], "fps": outro["fps"],
                               "audio": False, "file_dur": outro.get("file_dur"), "tc": outro.get("tc", 0.0)}})
    return parts


def drop_sentences(items: list[dict], bad: set[int]) -> list[dict]:
    """The items without the given sentence numbers (a range is split around them)."""
    out = []
    for it in items:
        if it["kind"] != "talk":
            out.append(it)
            continue
        run: list[int] = []
        for i in list(range(it["a"], it["b"] + 1)) + [None]:
            if i is not None and i not in bad:
                run.append(i)
            elif run:
                out.append({**it, "a": run[0], "b": run[-1]})
                run = []
    return out


def dedupe_intervals(intervals, margin: float = 0.05, min_len: float = 0.5) -> list[tuple[float, float]]:
    """No moment of the footage plays twice: whatever an earlier piece already uses is cut out of the later ones."""
    used: list[tuple[float, float]] = []
    out: list[tuple[float, float]] = []
    for a, b in intervals:
        pieces = [(a, b)]
        for ua, ub in used:
            nxt = []
            for x, y in pieces:
                if y <= ua - margin or x >= ub + margin:
                    nxt.append((x, y))
                    continue
                if x < ua - margin:
                    nxt.append((x, ua))
                if y > ub + margin:
                    nxt.append((ub, y))
            pieces = nxt
        for x, y in pieces:
            if y - x >= min_len:
                out.append((x, y))
                used.append((x, y))
    return out


def no_repeats(items: list[dict], sents: list[M.Sentence]) -> list[dict]:
    """The same line said twice (a retake, or the hook used again later) stays only where it is heard first: a whole passage that repeats
    one heard before is dropped, and inside a longer passage a single sentence (5+ words) that repeats an earlier one is cut out."""
    import difflib
    by = {s.idx: s for s in sents}
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()          # noqa: E731
    seen: list[str] = []
    said: list[str] = []
    out = []

    def repeat(text: str) -> bool:
        for t in said:
            if abs(len(t) - len(text)) <= 0.4 * len(text):
                sm = difflib.SequenceMatcher(None, text, t)
                if sm.real_quick_ratio() > 0.85 and sm.quick_ratio() > 0.85 and sm.ratio() > 0.85:
                    return True
        return False
    for it in items:
        if it["kind"] != "talk":
            out.append(it)
            continue
        text = norm(" ".join(by[i].text for i in range(it["a"], it["b"] + 1) if i in by))
        if len(text.split()) >= 4 and any(difflib.SequenceMatcher(None, text, t).ratio() > 0.8 for t in seen):
            continue
        seen.append(text)
        run: list[int] = []
        for i in list(range(it["a"], it["b"] + 1)) + [None]:
            line = norm(by[i].text) if i is not None and i in by else ""
            if i is not None and not (len(line.split()) >= 5 and repeat(line)):
                run.append(i)
                if line:
                    said.append(line)
                continue
            if run:
                out.append({**it, "a": run[0], "b": run[-1]})
                run = []
    return out


def keep_moving(cutaways: list[dict], cuts: list[dict], intervals, pool: list[dict], allowed, group: dict | None, out_len: float,
                exclude: set[int] = frozenset(), limit: float = 12.0, show: float = 3.2, choose=None,
                boundaries: list[float] | None = None, lenient: bool = True, jumps: list[float] | None = None,
                share: float = 0.6) -> tuple[list[dict], list[dict]]:
    """The twelve-second rule: never more than `limit` seconds of the same picture. A gap is broken up by a cutaway to B-roll / a photo, or by
    switching to the other camera (they take turns, so the video moves between all of them)."""
    def to_out(tm: float):
        acc = 0.0
        for a, b in intervals:
            if a - 1e-6 <= tm < b:
                return acc + tm - a
            acc += b - a
        return None

    def to_master(to: float):
        acc = 0.0
        for a, b in intervals:
            if to < acc + (b - a):
                return a + (to - acc)
            acc += b - a
        return None
    cutaways = [dict(c) for c in cutaways]
    cuts = [dict(c) for c in cuts]
    used = {c["scene"] for c in cutaways} | set(exclude)
    free = [i for i in sorted(range(len(pool)), key=lambda i: -pool[i]["quality"])
            if i not in used and (allowed is None or i in allowed) and pool[i]["end"] - pool[i]["start"] >= 1.5]

    # Step boundaries can join distant moments of an interview, and every jump cut inside one shot (a pause or a slip cut out) twitches.
    # Cover each join with a B-roll shot that starts a little before it and runs past it, the way an editor hides a cut; this is
    # deliberately separate from the cadence rule below because it protects the look of the cut, not just visual energy. Jumps come
    # biggest first and stop when B-roll would cover more than `share` of the video; with no B-roll left a shoot changes camera instead.
    def covered(t: float) -> bool:
        return any(c["at"] <= t - 0.3 and c["at"] + c["length"] >= t + 0.6 for c in cutaways)
    for kind, master_time in [("step", x) for x in boundaries or []] + [("jump", x) for x in jumps or []]:
        t = to_out(master_time)
        if t is None or t < 0.8 or t > out_len - 1.5 or covered(t):
            continue
        if kind == "jump" and sum(c["length"] for c in cutaways) >= share * out_len:
            break
        pick = (choose(t, free) if choose else free[0]) if free else None
        if pick is None and kind == "jump" and free and lenient:
            pick = free[0]
        if pick is not None:
            sc = pool[pick]
            at = max(0.0, t - 0.8)
            ln = min(show, out_len - at - 0.15, 1e9 if sc.get("photo") else sc["end"] - sc["start"])
            if ln >= 1.2 and at + ln >= t + 1.0:
                free.remove(pick)
                cutaways.append({"at": round(at, 2), "length": round(ln, 2), "src": round(sc["start"], 2), "scene": pick, "what": sc["what"],
                                 "why": "cover the jump between script steps" if kind == "step" else "cover the jump cut",
                                 "file": sc.get("file"), "photo": sc.get("photo")})
                cutaways.sort(key=lambda c: c["at"])
                continue
        if kind == "jump" and group and cuts:                              # no B-roll for it: another camera takes over at the cut
            seg = next((x for x in cuts if x["start"] < master_time - 0.05 and master_time < x["end"] - 0.5), None)
            if seg:
                alt = next((a["id"] for a in group["angles"] if a["id"] != seg["angle"]
                            and MC.cover(MC.angle_files(group, a["id"]), master_time) and MC.cover(MC.angle_files(group, a["id"]), seg["end"] - 0.05)), None)
                if alt:
                    cuts.append({"start": master_time, "end": seg["end"], "angle": alt})
                    seg["end"] = master_time
                    cuts.sort(key=lambda x: x["start"])
    cutaways.sort(key=lambda c: c["at"])

    def marks() -> list[float]:
        m = {0.0, out_len}
        for c in cutaways:
            m |= {c["at"], c["at"] + c["length"]}
        prev = None
        for s in sorted(cuts, key=lambda s: s["start"]):
            if prev is not None and s["angle"] != prev:
                o = to_out(s["start"])
                if o is not None:
                    m.add(o)
            prev = s["angle"]
        return sorted(m)
    broll_next = True
    for _ in range(80):
        m = marks()
        gap = next(((x0, x1) for x0, x1 in zip(m, m[1:]) if x1 - x0 > limit + 0.05), None)
        if not gap:
            break
        x0, x1 = gap
        t = x0 + min(4.5, (x1 - x0) - 3.2)
        done = False
        for mode in (("broll", "cam") if broll_next else ("cam", "broll")):
            if mode == "cam" and group and cuts:
                tm = to_master(t)
                seg = next((s for s in cuts if tm is not None and s["start"] <= tm < s["end"]), None)
                if seg and tm - seg["start"] >= 1.8 and seg["end"] - tm >= 1.8:
                    alt = next((a["id"] for a in group["angles"] if a["id"] != seg["angle"]
                                and MC.cover(MC.angle_files(group, a["id"]), tm) and MC.cover(MC.angle_files(group, a["id"]), seg["end"] - 0.05)), None)
                    if alt:
                        cuts.append({"start": tm, "end": seg["end"], "angle": alt})
                        seg["end"] = tm
                        cuts.sort(key=lambda s: s["start"])
                        done = True
            elif mode == "broll" and free:
                ln = round(min(show, x1 - t - 0.5), 2)
                nb = [c.get("file") for c in cutaways if abs(c["at"] - t) < 8]
                pick = choose(t, free) if choose else next((i for i in free if pool[i].get("file") not in nb), free[0])
                if pick is None and lenient:                                # nothing fits what is being said: still never leave the same picture for
                    pick = next((i for i in free if pool[i].get("file") not in nb), free[0])   # more than twelve seconds - show the best unused shot
                if pick is None:
                    continue
                free.remove(pick)
                sc = pool[pick]
                if ln >= 1.2:
                    cutaways.append({"at": round(t, 2), "length": min(ln, round(sc["end"] - sc["start"], 2)), "src": round(sc["start"], 2), "scene": pick,
                                     "what": sc["what"], "why": "never the same picture for more than twelve seconds", "file": sc.get("file"), "photo": sc.get("photo")})
                    cutaways.sort(key=lambda c: c["at"])
                    done = True
            if done:
                broll_next = not broll_next
                break
        if not done:
            break                                                          # nothing left to show (no spare footage, no other camera)
    return cutaways, cuts


def looks_like_logo(path) -> bool:
    """A logo is flat: a few colours, or see-through. A photograph (a video thumbnail, a still) has thousands of colours."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGBA")
            im.thumbnail((96, 96))
            px = list(im.getdata())
        clear = sum(1 for p in px if p[3] < 40) / max(1, len(px))
        colours = len({(p[0] >> 4, p[1] >> 4, p[2] >> 4) for p in px if p[3] >= 40})
        return clear > 0.15 or colours <= 40
    except Exception:  # noqa: BLE001 - can't look: don't put it on screen
        return False


def joined(a: dict, b: dict) -> bool:
    """True if piece b carries on the SAME sound as piece a, with no jump."""
    x, y = a.get("a"), b.get("a")
    return bool(x and y and x["path"] == y["path"] and abs(x["ss"] + a["d"] - y["ss"]) < 0.05)


def cutaways_of_parts(parts: list[dict], files: list[dict]) -> list[dict]:
    """The B-roll shots Final Cut gets: exactly the ones the render PLAYED (a planned cutaway that the render skipped because the clip ended too soon
    must not be in the project either - it pointed past the end of the media: "an edit with no respective media"), each kept inside its file."""
    out: list[dict] = []
    for p in parts:
        v = p["v"]
        if not v.get("cut") or not p.get("d"):
            continue
        last = out[-1] if out else None
        if last and last["path"] == v["path"] and last.get("layout") == v.get("layout") and abs(last["at"] + last["len"] - p["out"]) < 0.02 \
                and abs(last["start"] + last["len"] - v["ss"]) < 0.05:
            last["len"] += p["d"]                                          # a cutaway the render cut into pieces is one shot again
            continue
        bf = next((f for f in files if f["path"] == v["path"]), None)
        if bf is None:
            continue
        out.append({"at": p["out"], "len": p["d"], "path": bf["path"], "start": v["ss"], "w": bf["w"], "h": bf["h"], "fps": bf["fps"],
                    "tc": bf.get("tc", 0.0), "tc_df": bf.get("tc_df", False), "file_dur": bf.get("dur"),
                    **({"layout": v["layout"]} if v.get("layout") and (p.get("v0") or {}).get("path") else {})})
    keep = []
    for c in out:
        fd = c.get("file_dur")
        if fd:
            c["len"] = min(c["len"], fd - c["start"] - 0.05)
        if c["len"] >= 0.3 and c["start"] >= 0:
            keep.append(c)
    return keep


FILLER_WORDS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mmm", "hm"}
STUTTER_WORDS = {"i", "the", "a", "an", "and", "to", "of", "in", "it", "that", "we", "you", "my", "is", "was", "he", "she", "they", "so", "but",
                 "because", "if", "its", "i'm", "i've", "it's", "that's", "there"}


def filler_spans(words: list[dict]) -> list[tuple[float, float]]:
    """Where the speaker says "um" / "uh" and where a short word is stuttered ("I I", "the the": the first one goes, the second stays)."""
    norm = lambda w: re.sub(r"[^a-z']", "", str(w.get("text", "")).lower().replace("\u2019", "'"))          # noqa: E731
    spans = []
    for k, w in enumerate(words):
        t, dur = norm(w), w["end"] - w["start"]
        if t in FILLER_WORDS and 0.08 <= dur <= 1.6:
            spans.append((w["start"], w["end"]))
        elif t in STUTTER_WORDS and dur < 0.6 and k + 1 < len(words) and norm(words[k + 1]) == t and 0 <= words[k + 1]["start"] - w["end"] < 0.35:
            spans.append((w["start"], words[k + 1]["start"]))
    toks = [norm(w) for w in words]                                                   # a retake: the same few words said twice ("they get turned into content and they get turned into content"): the first try goes
    k = 0
    while k < len(words):
        for n in range(8, 2, -1):
            hit = None
            for c in (0, 1):                                                          # (c = one joining word between the two tries: "and", "so", "like"...)
                if c and (k + n >= len(words) or toks[k + n] not in ("and", "so", "but", "like", "then", "or")):
                    continue
                j = k + n + c
                if j + n <= len(words) and all(toks[k + i] and toks[k + i] == toks[j + i] for i in range(n)) and 0 <= words[j]["start"] - words[j - 1]["end"] < 1.0 \
                        and sum(len(t) for t in toks[k:k + n]) >= 10 and len(set(toks[k:k + n])) >= 3:
                    hit = j
                    break
            if hit is not None:
                spans.append((words[k]["start"], words[hit]["start"]))
                k = hit - 1
                break
        k += 1
    return spans


def cut_fillers(intervals: list[tuple[float, float]], words: list[dict], margin: float = 0.025, min_piece: float = 0.3) -> list[tuple[float, float]]:
    """The kept stretches without the filler sounds and stutters inside them (what Descript / Gling / TimeBolt do first). A filler is left alone when
    cutting it would strand a sliver (a piece shorter than min_piece) of speech beside it."""
    spans = [(a + margin, b - margin) for a, b in filler_spans(words) if b - a > 2 * margin]
    out: list[tuple[float, float]] = []
    for a, b in intervals:
        pieces = [(a, b)]
        for sa, sb in spans:
            if sb <= a or sa >= b:
                continue
            nxt = []
            for x, y in pieces:
                if sb <= x or sa >= y:
                    nxt.append((x, y))
                    continue
                left, right = sa - x, y - sb
                if (0.05 < left < min_piece) or (0.05 < right < min_piece):
                    nxt.append((x, y))                                       # would leave a sliver: keep this filler
                    continue
                if left >= min_piece:
                    nxt.append((x, sa))
                if right >= min_piece:
                    nxt.append((sb, y))
            pieces = nxt
        out += pieces
    return out


def punch_ins(parts: list[dict], zoom: float = 1.12) -> list[dict]:
    """Hide jump cuts the way editors do (Submagic / CapCut call it auto-zoom): when the same camera jumps to a later moment, the picture changes
    only slightly, and it looks like a mistake. Every second such cut is a slight punch-in (the picture scaled up ~12%, framed on the centre), so
    the cut reads as a deliberate change of shot. Camera switches and B-roll are already a visible change and are left alone; never two zoomed
    pieces in a row."""
    last_zoomed = False
    for k in range(1, len(parts)):
        prv, p = parts[k - 1], parts[k]
        if p["v"].get("cut") or prv["v"].get("cut") or p["v"]["path"] != prv["v"]["path"] or joined(prv, p):
            last_zoomed = False if (p["v"].get("cut") or p["v"]["path"] != prv["v"]["path"]) else last_zoomed
            continue
        if last_zoomed or prv.get("zoom"):
            last_zoomed = False
            continue
        p["zoom"] = zoom
        p["item"] = {**p["item"], "zoom": zoom}
        last_zoomed = True
    return parts


def seal_joins(parts: list[dict]) -> list[dict]:
    """Make the sound run on WITHOUT A BREAK wherever it is meant to: when a piece carries on the same sound as the one before it (another camera
    angle or a B-roll cutaway over the same speech), its sound must start exactly where the previous piece's sound ENDED. quantize() gives every
    piece a whole number of frames, so a piece's length can differ from the original by up to a frame, while the next piece still starts at its
    original place - the sound then repeats or skips up to one frame (about 40 ms): the little pop at an angle switch, in the render and in
    Final Cut alike. Here the next piece is moved to where the last one stopped (its picture along with it, so lip-sync is kept; the move is the
    rounding left over by quantize, never more than half a frame in total)."""
    for k in range(1, len(parts)):
        prv, p = parts[k - 1], parts[k]
        if not joined(prv, p):
            continue
        delta = prv["a"]["ss"] + prv["d"] - p["a"]["ss"]
        if abs(delta) < 1e-9 or abs(delta) > 0.06:
            continue
        p["a"] = {**p["a"], "ss": p["a"]["ss"] + delta}
        if not p["v"].get("cut"):                                           # the speaker's own picture moves with the sound (B-roll stays put)
            p["v"] = {**p["v"], "ss": p["v"]["ss"] + delta}
        elif p.get("v0"):                                                   # (under a B-roll: the speaker window of a layout moves instead)
            p["v0"] = {**p["v0"], "ss": float(p["v0"].get("ss") or 0.0) + delta}
        it = dict(p["item"])
        if it.get("kind") == "mc":
            it["t"] = it["t"] + delta
        elif it.get("kind") == "clip":
            it["start"] = it["start"] + delta
            if it.get("aud"):
                it["aud"] = {**it["aud"], "start": it["aud"]["start"] + delta}
        p["item"] = it
    return parts


def smooth_angles(parts: list[dict], group: dict | None, min_s: float = 4.0) -> list[dict]:
    """A camera angle that would show for less than `min_s` seconds (a flash of the other camera) takes the angle of the piece before it,
    or after it. Never a flicker."""
    if not group:
        return parts
    ids = {p["item"]["video"] for p in parts if p["item"].get("kind") == "mc"}
    if len(ids) < 2:
        return parts
    def runs():
        out, cur = [], None
        for k, p in enumerate(parts):
            ang = p["item"].get("video") if p["item"].get("kind") == "mc" and not p["v"].get("cut") else None
            if cur and cur["ang"] == ang and ang is not None:
                cur["end"] = k + 1
                cur["d"] += p["d"]
            else:
                cur = {"ang": ang, "start": k, "end": k + 1, "d": p["d"]}
                out.append(cur)
        return out
    for _ in range(4):
        changed = False
        rs = runs()
        for i, r in enumerate(rs):
            if r["ang"] is None or r["d"] >= min_s:
                continue
            for nb in (rs[i - 1] if i else None, rs[i + 1] if i + 1 < len(rs) else None):
                if not nb or nb["ang"] is None or nb["ang"] == r["ang"]:
                    continue
                ok = []
                for k in range(r["start"], r["end"]):
                    t, d = parts[k]["item"]["t"], parts[k]["d"]
                    c = MC.cover(MC.angle_files(group, nb["ang"]), t)
                    ok.append(c if c and MC.cover(MC.angle_files(group, nb["ang"]), t + d - 0.03) else None)
                if all(ok):
                    for k, c in zip(range(r["start"], r["end"]), ok):
                        parts[k]["v"] = {**parts[k]["v"], "path": c["path"], "ss": parts[k]["item"]["t"] - c["t0"]}
                        parts[k]["item"] = {**parts[k]["item"], "video": nb["ang"]}
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return parts


def quantize(parts: list[dict], fps: float) -> list[dict]:
    """Make every piece a whole number of frames (the running total never drifts more than half a frame) and drop pieces too short to
    hold a single frame. A piece with no picture in it would throw the joined video out of step with its sound."""
    # slivers (a piece that is only a frame or two, left where one file ends and the next begins) can't be cut reliably: they join the
    # piece before them when the sound carries straight on, else they are dropped
    merged: list[dict] = []
    for p in parts:
        if p["d"] < 0.15 and merged:
            prev = merged[-1]
            if prev["a"] and p["a"] and prev["a"]["path"] == p["a"]["path"] and abs(prev["a"]["ss"] + prev["d"] - p["a"]["ss"]) < 0.06:
                merged[-1] = {**prev, "d": prev["d"] + p["d"]}
            continue
        if p["d"] < 0.15 and not merged:
            continue
        merged.append(p)
    parts = merged
    out: list[dict] = []
    rate = Fraction(fps).limit_denominator(1001)
    t_target, frames_done = 0.0, 0
    for p in parts:
        t_target += p["d"]
        n = round(t_target * float(rate)) - frames_done
        if n < 1:
            continue
        frames_done += n
        q = {**p, "d": float(Fraction(n) / rate), "n": n}
        out.append(q)
    pos = 0.0
    for q in out:
        q["out"] = pos
        pos += q["d"]
    return out


def timeline_rate(fps) -> Fraction:
    """The rough cut's (and so the Final Cut timeline's) frame rate: the camera's own, as the exact standard rate (23.976 = 24000/1001,
    never rounded to 24). Above 60 fps (slow motion) the timeline runs at half; below 23 at 24."""
    from .polish import STANDARD_RATES
    f = float(fps or 30.0)
    while f > 61:
        f /= 2
    if f < 23:
        return Fraction(24)
    best = min(STANDARD_RATES, key=lambda r: abs(float(r) - f))
    return best if abs(float(best) - f) / float(best) < 0.002 else Fraction(f).limit_denominator(1001)


def _slice(p: dict, off: float, d: float) -> dict:
    """The part of piece `p` that starts `off` seconds into it and lasts `d` seconds."""
    q = {**p, "d": d, "out": p["out"] + off, "v": {**p["v"], "ss": p["v"]["ss"] + off},
         **({"v0": {**p["v0"], "ss": float(p["v0"].get("ss") or 0.0) + off}} if p.get("v0") else {}),
         "a": ({**p["a"], "ss": p["a"]["ss"] + off} if p["a"] else None), "item": dict(p["item"])}
    if q["item"]["kind"] == "mc":
        q["item"]["t"] += off
    else:
        q["item"]["start"] += off
        if q["item"].get("aud"):
            q["item"]["aud"] = {**q["item"]["aud"], "start": q["item"]["aud"]["start"] + off}
    return q


def fcp_items(parts: list[dict]) -> list[dict]:
    """Pieces -> the items of the Final Cut timeline (neighbouring pieces from the same angle/file are one clip)."""
    items: list[dict] = []
    pos = 0.0
    for p in parts:
        it = p["item"]
        last = items[-1] if items else None
        if it["kind"] == "mc":
            if last and last["kind"] == "mc" and last["video"] == it["video"] and last["audio"] == it["audio"] and abs(last["t"] + last["dur"] - it["t"]) < 0.05 \
                    and last.get("zoom") == it.get("zoom"):
                last["dur"] += p["d"]
            else:
                items.append({"kind": "mc", "out": pos, "dur": p["d"], "t": it["t"], "video": it["video"], "audio": it["audio"],
                              **({"zoom": it["zoom"]} if it.get("zoom") else {})})
        else:
            if last and last["kind"] == "clip" and last["path"] == it["path"] and abs(last["start"] + last["dur"] - it["start"]) < 0.05 \
                    and last.get("zoom") == it.get("zoom"):
                last["dur"] += p["d"]
            else:
                items.append({"kind": "clip", "out": pos, "dur": p["d"], "path": it["path"], "start": it["start"], "w": it["w"], "h": it["h"],
                              "fps": it["fps"], "audio": it["audio"], "file_dur": it.get("file_dur"), "tc": it.get("tc", 0.0),
                              "tc_df": it.get("tc_df", False), **({"aud": it["aud"]} if it.get("aud") else {}), **({"zoom": it["zoom"]} if it.get("zoom") else {})})
        pos += p["d"]
    return items



HEDGE_RE = re.compile(r"(not (entirely |really |exactly )?sure|i don'?t (really )?(know|remember)|i guess|i think maybe|how long we'?ve been|"
                      r"honestly,? i|can'?t remember|let me think|what was (the|that)|sorry,? what|you know what i mean)", re.I)
AD_RE = re.compile(r"\b(ad|ads|advert\w*|commercial|promo\w*)\b", re.I)
OUTRO_S = 2.6


NEEDS_MORE = re.compile(r"^(and|but|so|or|because|cause|the|a|an|to|of|for|with|that|which|who|if|when|then|like|is|was|are|were|my|our|"
                        r"your|their|his|her|its|this|these|those|i|we|you|they|he|she|in|on|at|from|about|into|as|than|not|just|really|"
                        r"very|going|gonna|get|got|um|uh)$", re.I)


def finish_sentence(items: list[dict], sents: list[M.Sentence], max_add: int = 4, max_s: float = 25.0) -> list[dict]:
    """The last thing said must be a finished thought. With punctuation (the speaker's own sentence ends) the last passage runs on until
    a sentence really ends - or, if that is too far, is cut back to the last sentence that did end. Without punctuation (older
    transcripts) a passage is unfinished when the next words follow with no pause (a sentence was split only by length) or it stops on a
    word that needs more ('and', 'the', 'to'...): it then runs on, but never across a real pause."""
    by = {s.idx: s for s in sents}
    punct = len(sents) >= 5 and sum(1 for s in sents if M.END.search(s.text)) >= 0.3 * len(sents)

    def unfinished(k: int) -> bool:
        s, nxt = by[k], by.get(k + 1)
        if punct:
            return not M.END.search(s.text)
        last = (s.text.split() or [""])[-1].strip(",.;:!?'\"")
        return (nxt is not None and nxt.start - s.end < 0.35) or bool(NEEDS_MORE.match(last))
    for it in reversed(items):
        if it["kind"] != "talk":
            continue
        b, added = it["b"], 0.0
        while (b in by and unfinished(b) and b + 1 in by and by[b + 1].start - by[b].end < 2.0 and b - it["b"] < max_add
               and added + by[b + 1].end - by[b + 1].start <= max_s):
            b += 1
            added += by[b].end - by[b].start
        goes_on = b + 1 in by and by[b + 1].start - by[b].end < 2.0          # (the speaker stopped there: that IS the end of the thought)
        if punct and b in by and unfinished(b) and goes_on:                 # it goes on, too far to take: end on the last sentence that ended
            done = [k for k in range(it["a"], it["b"] + 1) if k in by and not unfinished(k)]
            b = done[-1] if done else b
        it["b"] = b
        break
    return items


PROMO_RE = re.compile(r"\b(ad|ads|advert\w*|promo\w*|commercial|upbeat|hype\w*|punchy|fast[- ]?paced|reel|teaser|trailer|sizzle)\b", re.I)
KINDS = ("ad", "promo", "story", "business", "youtube")
# How each kind of video is edited. punch_ins: scale-ups that hide jump cuts (energetic: only YouTube / promo). graphics_per_s: at most one graphic per this
# many seconds. kinds: which graphics it may have. cut_per_s: how often the picture changes (B-roll). pace: the cut-pause default when you left "Natural".
PROFILES = {
    "story": {"label": "story or interview", "punch_ins": False, "graphics_per_s": 60, "kinds": ["animation", "recap", "fullquote", "chapter", "statement", "lower", "quote", "stat", "entity", "timeline"], "entities_per_s": 30, "cut_per_s": 18.0, "pace": None,
              "tone": "restrained and human: a name, a pull quote, now and then a number"},
    "business": {"label": "business video", "punch_ins": False, "graphics_per_s": 20, "kinds": ["animation", "search", "twocol", "ledger", "recap", "fullslide", "bigstat", "chapter", "statement", "lower", "slide", "chart", "stat", "quote", "entity", "chips", "compare", "timeline", "hub"], "entities_per_s": 22, "cut_per_s": 18.0,
                 "pace": None, "tone": "professional, clear and informative: slides for lists, charts for numbers, calm"},
    "youtube": {"label": "YouTube video", "punch_ins": True, "graphics_per_s": 11, "kinds": ["animation", "search", "twocol", "ledger", "recap", "fullslide", "bigstat", "fullquote", "chapter", "statement", "stat", "quote", "slide", "chart", "lower", "entity", "chips", "compare", "timeline", "hub"], "entities_per_s": 13, "cut_per_s": 10.0,
                "pace": "tight", "tone": "energetic and visual: a graphic to land every big point"},
    "promo": {"label": "promo or reel", "punch_ins": True, "graphics_per_s": 10, "kinds": ["bigstat", "fullquote", "statement", "stat", "quote", "slide", "lower", "entity", "chips", "compare"], "entities_per_s": 11, "cut_per_s": 6.0, "pace": "tight",
              "tone": "fast and punchy"},
    "ad": {"label": "advertisement", "punch_ins": False, "graphics_per_s": 16, "kinds": ["stat", "quote", "slide", "lower", "entity", "chips", "compare"], "entities_per_s": 10, "cut_per_s": 3.0, "pace": "tight",
           "tone": "short selling points"},
}
KIND_SCHEMA = {"type": "object", "properties": {"kind": {"type": "string", "enum": list(KINDS)}, "why": {"type": "string"}}, "required": ["kind"]}
KIND_SYSTEM = (
    "You read a video creator's notes and say what KIND of video they want, because each is edited differently. 'ad' = an advertisement "
    "that sells a business, product or service (mostly B-roll, short confident selling lines, an end card). 'promo' = an upbeat promo, reel, "
    "teaser or trailer built from fast punchy soundbites, but not selling a business. 'story' = an interview, testimony, vlog, documentary "
    "or talk told in order. 'business' = a calm, professional video for an organisation that is NOT an advertisement: a company or nonprofit overview, "
    "a client testimonial, an explainer, training or an internal update. 'youtube' = a video made for YouTube: a creator's vlog, commentary, tutorial, "
    "review or talking-head video that should feel energetic. Read the meaning, not keywords: 'don't make this feel like an ad', 'his promotion at work' "
    "or 'upbeat music under her story' are stories. When unsure, 'story'.")


def video_kind(ai, notes: str, chosen: str | None = None) -> str:
    """'ad' | 'promo' | 'story'. The form's choice wins; otherwise the AI reads the notes; the keyword guess is only the fallback."""
    if chosen in KINDS:
        return chosen
    notes = (notes or "").strip()
    if not notes:
        return "story"
    try:
        k = str(ai.complete_json("video_kind", KIND_SYSTEM, notes[:3000], KIND_SCHEMA).data.get("kind", "")).lower()
        if k in KINDS:
            return k
    except Exception as e:  # noqa: BLE001 - the keyword guess below
        log.warning(f"telling what kind of video this is failed ({e}); guessing from keywords")
    return "ad" if AD_RE.search(notes) else ("promo" if PROMO_RE.search(notes) else "story")
GRAPHIC_RE = re.compile(r"\b(graphics?|callouts?|key points?|lower[- ]?thirds?|on[- ]screen text|text on screen)\b", re.I)
POINT_SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "items": {"type": "object", "properties": {
    "at": {"type": "number"}, "text": {"type": "string"}}, "required": ["at", "text"]}}}, "required": ["points"]}


def key_points(ai, sents: list[M.Sentence], brief: str, out_len: float) -> list[dict]:
    """Short on-screen text for the key points as they are said: [{"at": output seconds, "len", "text"}]."""
    if not sents or out_len < 12:
        return []
    cap = max(1, min(8, int(out_len / 9)))
    lines = [f"  [{s.start:.1f}-{s.end:.1f}] {s.text}" for s in sents]
    prompt = ((f"The creator's notes: {brief}\n" if brief else "") + "Transcript of the finished video (timeline seconds):\n" + "\n".join(lines[:300]) +
              f"\n\nChoose up to {cap} KEY POINTS to show as short on-screen text (2 to 5 words each, title case, no punctuation, e.g. "
              "'Personalized Fittings'). 'at' = the second the point starts being said. Skip the first 4 seconds. One per point, spread out.")
    try:
        res = ai.complete_json("key_points", "You are a motion-graphics editor choosing the on-screen text for a video's key points.", prompt, POINT_SCHEMA)
    except Exception as e:  # noqa: BLE001 - no graphics is a fine result
        log.warning(f"choosing key points failed: {e}")
        return []
    out: list[dict] = []
    for p in sorted(res.data.get("points") or [], key=lambda p: float(p.get("at", 0))):
        try:
            at, text = float(p["at"]), " ".join(str(p["text"]).split())[:48]
        except (KeyError, TypeError, ValueError):
            continue
        if text and 4.0 <= at <= out_len - 2.5 and (not out or at >= out[-1]["at"] + out[-1]["len"] + 1.5) and len(out) < cap:
            out.append({"at": round(at, 2), "len": 2.8, "text": text})
    return out



def loudness_pass(path: Path, target: float) -> float | None:
    """Make the finished video as loud as it should be (a promo is louder than a story). Measured, then one clean gain change with a
    limiter so nothing clips; the picture is copied untouched. Returns the loudness it measured before."""
    import subprocess
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "info", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=1800)
    m = re.findall(r"\bI:\s+(-?\d+(?:\.\d+)?)\s+LUFS", r.stderr)
    if not m:
        return None
    measured = float(m[-1])
    gain = target - measured
    if abs(gain) < 0.7 or measured < -60:
        return measured
    tmp = path.with_name(path.stem + ".norm.mp4")
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(path), "-c:v", "copy", "-af",
                f"volume={gain:.2f}dB,alimiter=limit=0.89:level=disabled", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)],
               timeout=1800, what="setting the loudness")
    tmp.replace(path)
    return measured

def freezes(path: Path, min_s: float = 3.0) -> list[tuple[float, float]]:
    """[(start, end)] of every stretch where the picture doesn't change for min_s seconds or more."""
    import subprocess
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "info", "-i", str(path), "-vf", f"scale=320:-2,freezedetect=n=-70dB:d={min_s}", "-an",
                        "-f", "null", "-"], capture_output=True, text=True, timeout=1800)
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", r.stderr)]
    end_of_file = ffmpeg.duration(path) if len(ends) < len(starts) else 0.0
    return [(s, ends[k] if k < len(ends) else end_of_file) for k, s in enumerate(starts)]


def check_video(path: Path, expected: float, freeze: bool = True, reference: Path | None = None) -> list[str]:
    """What is wrong with a finished video (empty list = fine). `reference` = the rough cut it was finished from: a freeze that is
    in the rough cut as well (a still photo, a static screen recording) is the footage itself and is not a fault."""
    problems: list[str] = []
    try:
        info = ffmpeg.probe(path)
    except Exception as e:  # noqa: BLE001
        return [f"it can't be read ({e})"]
    vs, au = ffmpeg.video_stream(info), ffmpeg.audio_stream(info)
    if not vs:
        return ["it has no picture"]
    dv = float(vs.get("duration") or info.get("format", {}).get("duration") or 0)
    da = float(au.get("duration") or 0) if au else None
    if da is not None and abs(dv - da) > 0.6:
        problems.append(f"the picture is {dv:.1f} s but the sound is {da:.1f} s")
    if expected and abs(dv - expected) > max(1.0, 0.03 * expected):
        problems.append(f"it is {dv:.1f} s long but should be {expected:.1f} s")
    if freeze and dv > 5:
        found = freezes(path)
        if found and reference is not None and Path(reference).exists():
            ref = freezes(Path(reference), 2.0)                      # the same moment frozen in the rough cut = what was filmed
            found = [(s, e) for s, e in found if not any(rs <= s + 0.5 and e - 0.5 <= re_ for rs, re_ in ref)]
        if found:
            problems.append(f"the picture freezes (from {found[0][0]:.1f} s)")
    return problems

REVISE_SYSTEM = (
    "You turn a video creator's spoken-style feedback on a finished video into precise editing operations. Read their whole message "
    "and fill in EVERY operation they asked for (people ask for several things at once). Leave anything they did not ask for at its "
    "neutral value. Rules: 'slightly too long / a bit long' with no number = shorter_pct 10; 'too long / cut it down' = 20; 'much "
    "shorter / way too long' = 35. If they give a time ('cut 20 seconds', 'make it a minute'), use shorter_by_s or target_total_s. "
    "'Switch / change / swap the song' = change_music true (music_request holds a style only if they named one, e.g. 'something "
    "more upbeat'). 'Cut to the beat / bend to the beat / on the beat / sync to the music' = beat_sync 'on'; if they also changed the "
    "song, the beat means the NEW song. 'Add a (better) hook that says XYZ' = hook_text is exactly XYZ in their words (fix only "
    "obvious typos); 'a better hook' with no words given = better_hook true and hook_text empty. Anything about what is said or "
    "shown (cut this, move that line to the front, remove the part about X) goes in content_instruction in their own words. "
    "'summary' = one short sentence listing what you will do.")
REVISE_SCHEMA = {"type": "object", "properties": {
    "shorter_pct": {"type": "number"}, "shorter_by_s": {"type": "number"}, "longer_by_s": {"type": "number"},
    "target_total_s": {"type": "number"}, "change_music": {"type": "boolean"}, "music_request": {"type": "string"},
    "beat_sync": {"type": "string", "enum": ["on", "off", "unchanged"]}, "hook_text": {"type": "string"},
    "better_hook": {"type": "boolean"}, "move_to_front_quote": {"type": "string"}, "content_instruction": {"type": "string"},
    "summary": {"type": "string"}},
    "required": ["shorter_pct", "shorter_by_s", "longer_by_s", "target_total_s", "change_music", "music_request", "beat_sync",
                 "hook_text", "better_hook", "move_to_front_quote", "content_instruction", "summary"]}


LEN_SCHEMA = {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}
_WORDNUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30,
            "half a": 0.5}


def length_from_notes(ai, notes: str) -> float:
    """How long the creator said the video should be ('a minute long', 'under 90 seconds', 'around 5 min'); 0 = they didn't say."""
    notes = (notes or "").strip()
    if not notes:
        return 0.0
    guess = 0.0
    m = re.search(r"(?:\b(half a|a|one|two|three|four|five|six|ten|fifteen|twenty|thirty)\b|(\d+(?:\.\d+)?))\s*[- ]?(minutes?|mins?|seconds?|secs?)\b", notes, re.I)
    if m:
        n = float(m.group(2)) if m.group(2) else _WORDNUM.get(m.group(1).lower(), 1)
        guess = n * (60 if m.group(3).lower().startswith("min") else 1)
    try:
        res = ai.complete_json("video_length", "You read a video creator's notes and find the length they want. Answer in seconds; 0 if they "
                               "did not ask for a length. 'a minute long' = 60; 'under 2 minutes' = 110; 'around five minutes' = 300. "
                               "Ignore times that are positions in the footage (e.g. 'at 3:20').", notes[:3000], LEN_SCHEMA)
        sec = float(res.data.get("seconds") or 0)
        return sec if 5 <= sec <= 7200 else (guess if 5 <= guess <= 7200 else 0.0)
    except Exception as e:  # noqa: BLE001 - the pattern match above is the fallback
        log.warning(f"reading the length from the notes failed ({e}); pattern guess {guess}")
        return guess if 5 <= guess <= 7200 else 0.0


def parse_request(ai, text: str, plan: dict, cur_len: float) -> dict:
    state = (f"The video now: {cur_len:.0f} seconds long. Music: {(plan.get('music') or {}).get('title') or 'none yet'}. "
             f"Cutting to the beat: {'yes' if plan.get('beat_sync') else 'no'}. On-screen hook: {plan.get('hook_text') or '(none)'}.")
    res = ai.complete_json("revise_request", REVISE_SYSTEM, f"{state}\n\nThe creator says:\n{text}", REVISE_SCHEMA)
    ops = dict(res.data)
    log.info(f"revision understood as: {ops}")
    return ops
