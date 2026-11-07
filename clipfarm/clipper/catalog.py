"""The footage catalog: every video and photo on a big drive, looked at ONCE, so it can be searched in plain words ("students walking
to class in the fall") and its shots used as B-roll in any edit. Everything runs on this Mac: no AI plan, no API, no tokens.

    Catalog(cfg)                         catalog.*: roots (the folders to catalogue), dir (where the catalog lives), model, vision_model
    .scan()                              what is on the drives, from the folder listing only (nothing is opened)
    .index(deadline, limit, progress)    pass 1, fast: a frame every few seconds -> a small thumbnail + a picture embedding (searchable)
    .describe(deadline, limit, progress) pass 2, slow: the local vision model says what each scene shows and how usable it is
    .search(text, n, kind, under)        -> shots, best match first
    .shots_for(ai, sentences, brief)     -> B-roll candidates for an interview: the AI writes what to look for, the catalog finds it
    .fcpxml(shots, name)                 -> a Final Cut event with the found shots as trimmed clips (media referenced where it is)

On disk (catalog.dir; nothing on the drives is ever written):
    files/<id[:2]>/<id>.json   one record per source file: path, size, date, length, frames, scenes (once described)
    thumbs/<id>/<k>.jpg        the frames, 384 px wide: what search results show and what the vision model reads (as contact sheets)
    emb/<id>.npy               one picture embedding per frame
    index.npz + index.json     every embedding in one matrix for searching, rebuilt when records change

A file whose size or date changed is looked at again; a file that is gone (while its drive is plugged in) is forgotten.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from ..core import ffmpeg
from . import cameramotion as CM
from . import tagwords as TW
from ..core.log import get_logger
from ..core.state import atomic_write_json

log = get_logger("clipper.catalog")
VIDEO_EXT = {".mp4", ".mov", ".mxf", ".m4v", ".mts", ".avi", ".mkv"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".dng", ".cr2", ".cr3", ".arw", ".nef", ".raf"}
RAW_EXT = {".dng", ".cr2", ".cr3", ".arw", ".nef", ".raf"}            # a RAW next to a JPEG of the same name is the same photo: the JPEG stands for it
EDIT_PHOTO_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}           # photos the editor can animate (ffmpeg reads these)
SKIP_NAMES = {"$recycle.bin", "system volume information", ".trashes", ".spotlight-v100", ".fseventsd", ".temporaryitems",
              "video effects", "lightroom local storage", "_lightroom_copy", "gear selling pics", "bills of sale",
              # finished results, not footage: exports carry captions and graphics, and our own Output/ is a finished video
              "output", "exports", "export", "renders", "render files", "final cut", "notes", ".aivideo", "rejected", "transcoded media",
              "proxy media", "optimized media", "cache", "trash"}
PENDING = "pending.json"                                  # a run asked for now, carried on after a restart (tick)
TRANSIENT = re.compile(r"no such file|input/output|timed out|resource busy|not permitted|picture model|retry", re.I)   # worth another try
MODEL = "google/siglip2-base-patch16-224"
THUMB_W = 384
LIVE_N = 24                      # what Find footage's live panel can show: the last few things the catalogue did
JPEG_VF = f"scale={THUMB_W}:-2:out_range=pc,format=yuvj420p"   # (the JPEG encoder refuses some cameras' limited-range video without this)
SHOT_MIN, SHOT_MAX = 4.0, 8.0                    # a found shot: the matching stretch, at least 4 s (where the clip allows), at most 8
STOP = {"with", "from", "that", "this", "their", "there", "they", "them", "into", "onto", "over", "some", "about", "while", "where",
        "people", "person", "shot", "footage", "video", "clip", "image", "photo", "picture", "showing", "shows"}

QUERY_SYSTEM = (
    "You plan B-roll for an interview video. A footage library is searched with short visual descriptions, so for each passage of the "
    "interview write what a camera could have filmed that shows what is being said: people, places, actions, objects, events. Each "
    "description is 3 to 10 words, concrete and literal (for example 'students studying together in a library', 'graduates throwing "
    "caps in the air', 'campus building on a sunny day', 'volleyball player spiking the ball'). Use the creator's notes for who and where. "
    "Give one or two per passage; give none where nothing visual fits. Never describe the speaker talking to the camera. Never use a "
    "person's name or an organisation's name: the search only sees pictures, so describe what is visible ('a young woman walking across "
    "a campus', not 'Elena walking across Northfield College').")
QUERY_SCHEMA = {"type": "object", "properties": {"passages": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "looks": {"type": "array", "items": {"type": "string"}}}, "required": ["n", "looks"]}}},
    "required": ["passages"]}
TILES_SYSTEM = (
    "You describe footage for a video editor who cannot see it. You get one contact sheet: numbered tiles, each a frame from a "
    "different shot. For every tile say in one concrete sentence what it shows (subject, action, setting) - no guessing about meaning - "
    "and rate it as B-roll from 1 to 10 (sharp, steady, well lit, interesting = high; blurry, dark, empty, accidental = low).")
TILES_SCHEMA = {"type": "object", "properties": {"tiles": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "what": {"type": "string"}, "quality": {"type": "number"}}, "required": ["n", "what", "quality"]}}},
    "required": ["tiles"]}


class NotTheFile(RuntimeError):
    """A failure that is not the file's fault (its drive went away, the picture model would not load): the file stays uncatalogued, to
    be looked at next time, and the run stops instead of writing off everything it has not reached yet."""


COMPILE_S = 45.0                         # during a run, the search index is refreshed at least this often


BACKGROUND = ["nice", "-n", "15"] if shutil.which("nice") else []       # (cataloguing yields to whatever you are doing in the app: a 4K shot playing was dropping frames)
FINISHED_DIR = re.compile(r"^(finals?|finished( videos?)?|exports?|exported|deliverables?|delivered|renders?|rendered|published|masters?)$", re.I)


# Named like an export only when the name ENDS that way ("Fall Recap FINAL", "Fall Recap v2", "Fall Recap (rough cut)"), so "Final Four game" and "V8 engine" stay footage.
FINISHED_NAME = re.compile(r"(?:[ \-(]|^)(?:finals?|export(?:ed)?|rough[ -]*cut|fine[ -]*cut|first[ -]*cut|master|rendered|final[ -]*export)(?:[ -]*v?\d{1,2})?\)?$"
                           r"|[ \-]v\d{1,2}\)?$|[ \-]draft(?:[ -]*\d{1,2})?\)?$", re.I)


def finished_video(rel: Path) -> bool:
    """True for a finished video, not footage: in a folder called Final/Exports/Delivered..., or named like an export (final, V2, rough cut,
    master...). A folder of camera files (raw, footage, B-roll...) is never judged by the file's name (localproj.earlier_edit)."""
    from .localproj import CAMERA_DIR
    if any(FINISHED_DIR.match(part) for part in rel.parts[:-1]):
        return True
    if any(CAMERA_DIR.search(part) for part in rel.parts[:-1]):
        return False
    return bool(FINISHED_NAME.search(re.sub(r"[_.]+", " ", rel.stem).strip()))


def rid(path: str | Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def words_of(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 4 and w not in STOP]


def in_window(window, now: datetime | None = None) -> bool:
    """True inside ["19:00", "07:00"] (a window may run past midnight). No window, or "always", = always."""
    if not window or window == "always":
        return True
    now = now or datetime.now()
    a, b = ([int(x) for x in str(t).split(":")] for t in window)
    m, ma, mb = now.hour * 60 + now.minute, a[0] * 60 + a[1], b[0] * 60 + b[1]
    return ma <= m < mb if ma <= mb else (m >= ma or m < mb)


class _AccountVision:
    """The catalogue's describer when the person chose their AI account: the same request, through LocalAI (Claude's quick model or the
    ChatGPT plan; offline, this Mac's picture model). About 2,000 tokens a file with Claude."""

    def __init__(self, cfg):
        from .localai import LocalAI
        self.ai = LocalAI(cfg)

    def complete_json(self, system, prompt, schema, images=None):
        return self.ai.complete_json("catalog_describe", system, prompt, schema, images=images)


# ---------------------------------------------------------------------------------------------- pictures -> numbers
class SigLIP:
    """Pictures and words in one space (SigLIP 2, on the Mac's GPU): the embedding of a sentence lands near the frames that look like it."""

    def __init__(self, name: str = MODEL):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "mps" else torch.float32
        try:                                                             # the copy on disk first: works offline, no update check
            self.model = AutoModel.from_pretrained(name, dtype=self.dtype, local_files_only=True).to(self.dev).eval()
            self.proc = AutoProcessor.from_pretrained(name, local_files_only=True)
        except OSError:                                                  # not downloaded yet (Setup does it): fetch it now
            self.model = AutoModel.from_pretrained(name, dtype=self.dtype).to(self.dev).eval()
            self.proc = AutoProcessor.from_pretrained(name)
        self.name = name
        self.scale, self.bias = float(self.model.logit_scale.exp()), float(self.model.logit_bias)
        self.lock = threading.Lock()                     # one thing at a time on the GPU (search requests and the indexer share it)

    def _norm(self, f):
        f = getattr(f, "pooler_output", f)
        return self.torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    def images(self, ims: list) -> "np.ndarray":  # noqa: F821
        import numpy as np
        out = []
        with self.lock, self.torch.no_grad():
            for k in range(0, len(ims), 32):
                x = self.proc(images=ims[k:k + 32], return_tensors="pt")
                out.append(self._norm(self.model.get_image_features(pixel_values=x["pixel_values"].to(self.dev, self.dtype))))
        return np.concatenate(out) if out else np.zeros((0, 768), "float32")

    def text(self, texts: list[str]) -> "np.ndarray":  # noqa: F821
        with self.lock, self.torch.no_grad():
            x = self.proc(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
            return self._norm(self.model.get_text_features(**x))

    def chance(self, score: float) -> float:
        """How likely the picture really shows the words (SigLIP's own sigmoid): ~0.1 is a clear match, below 0.001 is not."""
        z = score * self.scale + self.bias
        return 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))


# ---------------------------------------------------------------------------------------------- the catalog
class Catalog:
    def __init__(self, cfg, embedder=None, vision=None):
        c = cfg.get("catalog") or {}
        self.workers = int(c.get("workers") or 0)                 # files read at once: 0 = as many as the Mac can spare (loadgov.py), else exactly this many
        self.load_mode = str(c.get("load") or "adapt")           # "adapt" | "gentle" (one file at a time) | "full" (Settings > Footage library)
        self.governor = None                                     # the running pass's loadgov.Governor (the page's activity messages reach it)
        d = Path(os.path.expanduser(str(c.get("dir", "data/catalog"))))
        self.dir = d if d.is_absolute() else cfg.root / d
        self.roots = [Path(os.path.expanduser(str(r))) for r in (c.get("roots") or [])]
        self.skip = {str(s).lower() for s in (c.get("skip") or [])} | SKIP_NAMES
        self.every = float(c.get("frame_every_s", 3.0))
        self.max_frames = int(c.get("max_frames", 120))
        self.photos = bool(c.get("photos", True))
        self.model = str(c.get("model", MODEL))
        self.vision_model = str(c.get("vision_model") or cfg.get("llm.ollama.vision_model") or "qwen3-vl:8b")
        self.ollama_url = str(c.get("ollama_url") or cfg.get("llm.ollama.base_url") or "http://localhost:11434")
        self.min_score = float(c.get("min_score", 0.085))       # below this a frame does not really show the words (SigLIP 2 cosine)
        self.window = c.get("window")                            # automatic indexing hours, e.g. ["19:00", "07:00"], or "always"
        self.cfg = cfg
        self.describe_with = str(c.get("describe_with") or "local")   # "account": the person's AI account describes (Settings › Footage library)
        self._embedder, self._vision = embedder, vision
        self._idx = None
        self._idx_lock = threading.Lock()
        self._tagger = None                                      # tagwords.TagWords: three picture keywords per file (tagwords.py)
        self._tag_lock = threading.Lock()
        self.stop = threading.Event()
        self.status: dict = {"running": False, "stage": "", "done": 0, "total": 0, "error": None}
        self.thread: threading.Thread | None = None
        self.only: list[Path] | None = None                      # a run limited to some folders (start(only=...)), else every root
        self._run_lock = threading.Lock()                        # start(): one run at a time, even when two requests arrive together
        self._wanted: list[str] = []                             # first(): folders asked for while a hand-over is under way
        self._handing = False
        self._starting = False
        self._dirty = False
        self._emb_lock = threading.Lock()                        # (the first search and the indexer may both ask for the model at once)
        self.busy = None                                         # () -> True while a video is being made: the catalogue waits for it
        from collections import deque
        self.live: deque = deque(maxlen=LIVE_N)                  # Find footage's live panel (live_since): "now" and "done" events
        self.live_n = 0
        self._live_lock = threading.Lock()

    # -- what it is doing right now (Find footage's live panel) ----------------------------------
    def _live(self, kind: str, **data) -> None:
        """One event: "now" (the file in hand: id, rel, step "frames" | "describe", k = a frame to show) or "done" (a file finished:
        its frame, and for a description the first scenes and who described it). The stills are the catalogue's own frames
        (/api/library/thumb/<id>/<k>), so nothing extra is decoded or stored. Never raises: a panel never slows the catalogue."""
        try:
            with self._live_lock:
                self.live_n += 1
                self.live.append({"n": self.live_n, "kind": kind, "at": round(time.time(), 2), **data})
        except Exception:  # noqa: BLE001
            pass

    def live_since(self, after: int = 0) -> dict:
        """{"events": [...after n], "last": n, "status": {...}} for the page (GET /api/app/library/live/<after>)."""
        with self._live_lock:
            ev = [dict(e) for e in self.live if e["n"] > after]
            last = self.live_n
        return {"events": ev, "last": last,
                "status": {k: self.status.get(k) for k in ("running", "stage", "done", "total", "started", "error", "workers", "load")}}

    @staticmethod
    def who(model) -> str:
        """Who described a file, in plain words: "Claude Haiku", "ChatGPT", or "this Mac"."""
        m = str(model or "").lower()
        if "claude" in m or m in ("haiku", "sonnet", "opus"):
            fam = next((f for f in ("haiku", "sonnet", "opus") if f in m), "")
            return "Claude" + (" " + fam.title() if fam else "")
        if "gpt" in m or m.startswith(("o1", "o3", "o4", "chatgpt")):
            return "ChatGPT"
        return "this Mac"

    def _mid(self, rec: dict | None) -> int:
        return max(0, int((rec or {}).get("frames") or 1) // 2)

    def _wait(self) -> None:
        while self.busy and self.busy() and not self.stop.is_set():
            self.status["stage"] = "Waiting: a video is being made"
            time.sleep(15)

    def enabled(self) -> bool:
        return bool(self.roots)

    # -- storage --------------------------------------------------------------------------------
    def _rec_path(self, i: str) -> Path:
        return self.dir / "files" / i[:2] / f"{i}.json"

    def load(self, i: str) -> dict | None:
        try:
            return json.loads(self._rec_path(i).read_text())
        except (OSError, ValueError):
            return None

    def save(self, rec: dict) -> None:
        p = self._rec_path(rec["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(p)
        self._changed()

    def forget(self, i: str) -> None:
        self._rec_path(i).unlink(missing_ok=True)
        (self.dir / "emb" / f"{i}.npy").unlink(missing_ok=True)
        (self.dir / "motion" / f"{i}.npy").unlink(missing_ok=True)
        shutil.rmtree(self.dir / "thumbs" / i, ignore_errors=True)
        self._changed()

    def records(self):
        for p in sorted((self.dir / "files").glob("*/*.json")):
            try:
                yield json.loads(p.read_text())
            except (OSError, ValueError):
                continue

    def _changed(self) -> None:
        self._dirty = True

    def thumb(self, i: str, k: int) -> Path:
        return self.dir / "thumbs" / i / f"{int(k):04d}.jpg"

    # -- what is on the drives -------------------------------------------------------------------
    def mounted(self) -> list[Path]:
        """The folders this run covers that are plugged in: all of them, or only the ones a run was started for (self.only)."""
        return [r for r in (self.only or self.roots) if r.is_dir()]

    @staticmethod
    def under(path: str, root) -> bool:
        return str(path).startswith(str(root).rstrip("/") + "/")

    def _skipped(self, name: str) -> bool:
        from .localproj import SKIP_PATTERN
        n = name.lower()
        return n in self.skip or n.startswith(".") or bool(SKIP_PATTERN.search(name))

    def scan(self) -> list[dict]:
        """Every video (and photo) under the roots, from the folder listing only. Final Cut libraries, proxies, render files, the
        recycle bin and the folders in catalog.skip are left out. A RAW photo with a JPEG of the same name counts once (the JPEG)."""
        out, seen = [], set()
        errs: list[str] = []
        left_out = 0
        roots = self.mounted()                     # (a folder inside another is walked too, even when the walk would skip its name;
        for root in dict.fromkeys(roots):          #  the ids keep each file once)
            base = Path(*root.parts[:3]) if len(root.parts) > 2 and root.parts[1] == "Volumes" else root.parent   # names start at the drive
            for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: errs.append(f"{type(e).__name__}: {e}")):
                dirnames[:] = sorted(d for d in dirnames if not self._skipped(d))
                stems = {Path(f).stem.lower() for f in filenames if Path(f).suffix.lower() in EDIT_PHOTO_EXT | {".heic", ".heif"}}
                for f in sorted(filenames):
                    if f.startswith("."):
                        continue
                    ext = Path(f).suffix.lower()
                    kind = "video" if ext in VIDEO_EXT else "photo" if (ext in PHOTO_EXT and self.photos) else None
                    if not kind or (ext in RAW_EXT and Path(f).stem.lower() in stems):
                        continue
                    p = Path(dirpath) / f
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_size < 1024:
                        continue
                    if kind == "video" and finished_video(p.relative_to(root)):
                        left_out += 1
                        continue                                  # (an export or a final: edited pictures, with graphics and captions on them)
                    if rid(p) in seen:
                        continue
                    seen.add(rid(p))
                    out.append({"id": rid(p), "path": str(p), "rel": str(p.relative_to(base)), "root": str(root), "kind": kind,
                                "size": st.st_size, "mtime": int(st.st_mtime)})
        self.scan_info = {"found": len(out), "errors": errs[:5], "finished": left_out, "roots": [str(r) for r in dict.fromkeys(roots)]}
        return out

    def nothing_found(self) -> str:
        """Plain words for why a run over folders found no videos or photos to catalogue ("" when it found some)."""
        info = getattr(self, "scan_info", None) or {}
        if info.get("found"):
            return ""
        where = ", ".join(Path(r).name for r in info.get("roots") or []) or "that folder"
        errs = " ".join(info.get("errors") or [])
        if "Permission" in errs or "Operation not permitted" in errs:
            return (f"macOS won't let FirstCut read “{where}”. Open System Settings › Privacy & Security › Files & Folders (or Full Disk Access), "
                    "turn FirstCut on for Removable Volumes, then try again.")
        if errs:
            return f"Couldn't read “{where}”: {errs[:160]}"
        if info.get("finished"):
            return (f"Found no footage in “{where}”: its {info['finished']} video{'s' if info['finished'] != 1 else ''} look like finished edits "
                    "(they are in Final or Export folders, or named final, v2 or rough cut), which are left out.")
        return f"There are no videos or photos in “{where}” that can be catalogued (it may be empty, or only hold file types the catalogue doesn't read)."

    def todo(self, found: list[dict]) -> list[dict]:
        """The files not catalogued yet (or changed since, or catalogued with another model)."""
        out = []
        for f in found:
            r = self.load(f["id"])
            if not r or r.get("size") != f["size"] or r.get("mtime") != f["mtime"] or r.get("model") != self.model or \
                    (r.get("colour") is None and f["kind"] == "video" and r.get("frames") and self._lut(f["path"])) or \
                    (r.get("error") and TRANSIENT.search(str(r["error"]).replace(str(f.get("path") or "\0"), ""))):   # (the error, not the file's name)
                out.append(f)
        return out

    def prune(self, found: list[dict]) -> int:
        """Forget files that are gone - only under roots that are plugged in (an unplugged drive forgets nothing)."""
        live = {f["id"] for f in found}
        roots = self.mounted()                                           # (only the folders this run looked at)
        n = 0
        for r in list(self.records()):
            if r["id"] not in live and any(self.under(r.get("path", ""), root) for root in roots):
                self.forget(r["id"])
                n += 1
        return n

    # -- keywords: the three words that best describe a file's pictures (tagwords.py) --------------------------------------------
    def tagger(self):
        """The word matrix for the picture model that is LOADED (never loads or downloads it): None while there is no model yet."""
        with self._tag_lock:
            if self._tagger is None and self._embedder is not None:
                self._tagger = TW.TagWords(self.dir, self._embedder, self.model)
            return self._tagger

    def picture_tags(self, frames, rel: str = "") -> list[str]:
        """Up to three words for a file's frame embeddings; with no model (or no match) the words of its folders and file name."""
        try:
            tg = self.tagger()
            tags = tg.pick(frames) if tg is not None else []
        except Exception as e:  # noqa: BLE001 - keywords are a bonus, never a reason to fail a file
            log.info(f"keywords not made for {rel}: {e}")
            tags = []
        return tags or TW.path_words(rel)

    def fill_tags(self, limit: int = 600, ids=None) -> int:
        """Keywords for catalogued files that have none yet (made before keywords existed): from their embeddings on disk, no video is read.
        Only with the picture model already loaded (so it never downloads it). Returns how many were tagged."""
        import numpy as np
        if self.tagger() is None:
            return 0
        n = 0
        for rec in (self.load(i) for i in ids) if ids is not None else self.records():
            if not rec or rec.get("tags") is not None or not rec.get("frames") or rec.get("error"):
                continue
            try:
                e = np.load(self.dir / "emb" / f"{rec['id']}.npy").astype("float32")
            except (OSError, ValueError):
                continue
            self.save({**rec, "tags": self.picture_tags(e, rec.get("rel") or "")})
            n += 1
            if n >= limit:
                break
        if n and self._tagger is not None:
            self._tagger.flush()
        return n

    def _described_tags(self, rec: dict, texts: list[str]) -> list[str]:
        """The record's keywords after a description arrived: the picture words, sharpened by words the description uses that few others do."""
        tg = self.tagger()
        if tg is None:
            return rec.get("tags") or TW.path_words(rec.get("rel") or "")
        try:
            tg.note_description(" ".join(texts))
            return tg.merge(rec.get("tags") or [], texts)
        except Exception as e:  # noqa: BLE001
            log.info(f"keywords not sharpened for {rec.get('rel')}: {e}")
            return rec.get("tags") or []

    # -- pass 1: frames + embeddings ---------------------------------------------------------------
    def embedder(self):
        with self._emb_lock:
            if self._embedder is None:
                self._embedder = SigLIP(self.model)
            return self._embedder

    def _lut(self, path) -> str:
        """Log footage (S-Log3...) is seen in normal colour: search, the AI's descriptions and the previews (logcolor.py)."""
        from . import logcolor
        return logcolor.vf_prefix(path, getattr(self, "cfg", None))

    def _lut_after(self, path) -> str:
        """The same colour change, written to go after the scale: a 3D LUT on a 4K 10-bit frame is slow, on a 384 px one it is free."""
        f = self._lut(path).rstrip(",")
        return ("," + f) if f else ""

    def _grab(self, path: Path, step: float, dur: float, keyframes: bool, motion: list | None = None) -> list[bytes]:
        """The stills. With `motion` (a list) the same ffmpeg call also writes tiny gray frames of every decoded keyframe to a second output,
        for the camera-motion analysis (cameramotion.py): nothing is decoded twice. The list gets (times, frames)."""
        want_motion = keyframes and motion is not None
        with tempfile.TemporaryDirectory(prefix="cat-") as td:
            r = subprocess.run([*BACKGROUND, ffmpeg.bin_path("ffmpeg"), "-v", "info" if want_motion else "error", "-nostats",
                                *(["-skip_frame", "nokey"] if keyframes else []), "-i", str(path), "-an",
                                "-vf", f"fps=1/{step:.4f},scale={THUMB_W}:-2{self._lut_after(path)},{JPEG_VF}", "-q:v", "5", str(Path(td) / "%05d.jpg"),
                                *(CM.second_output() if want_motion else [])],
                               capture_output=True, timeout=max(300, dur * (3 if keyframes else 8)))
            if want_motion:
                try:
                    motion.append(CM.read_frames(r.stdout, r.stderr))
                except Exception as e:  # noqa: BLE001 - the stills are what matter; motion is a bonus
                    log.info(f"camera motion frames for {path.name}: {e}")
            return [p.read_bytes() for p in sorted(Path(td).glob("*.jpg"))]

    def _frames(self, path: Path, dur: float, motion: list | None = None) -> tuple[list[bytes], float]:
        """JPEG frames about every `every` s -> (frames, step). Keyframes only first: fast even for 4K camera files, which have one every
        half second or so. An export or a screen recording has one every 10 s or more: then the keyframe pass comes back short (the
        end missing) or repeats the same picture, and the file is decoded in full instead."""
        step = max(self.every, dur / max(1, self.max_frames))
        frames = self._grab(path, step, dur, keyframes=True, motion=motion)
        want = max(1, math.ceil(dur / step))
        if frames and (len(frames) < want - 1 or (len(frames) >= 3 and len(set(frames)) < 0.75 * len(frames))):
            full = self._grab(path, step, dur, keyframes=False)
            if len(full) >= len(frames):
                frames = full
        if not frames:                                          # (no keyframe decoded: try one ordinary frame from the middle)
            r = subprocess.run([*BACKGROUND, ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{dur / 2:.2f}", "-i", str(path), "-frames:v", "1",
                                "-vf", f"scale={THUMB_W}:-2{self._lut_after(path)},{JPEG_VF}", "-f", "mjpeg", "-q:v", "5", "-"], capture_output=True, timeout=300)
            frames = [r.stdout] if r.stdout else []
        return frames, step

    def _photo(self, path: Path) -> bytes | None:
        """A 384 px JPEG of a photo, the right way up (Pillow; RAW and HEIC through macOS's own `sips`)."""
        from PIL import Image, ImageOps
        if path.suffix.lower() in EDIT_PHOTO_EXT:
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((THUMB_W, THUMB_W * 2))
                    b = io.BytesIO()
                    im.save(b, "JPEG", quality=80)
                    return b.getvalue()
            except Exception:  # noqa: BLE001 - try sips below
                pass
        if shutil.which("sips"):
            with tempfile.TemporaryDirectory(prefix="cat-") as td:
                out = Path(td) / "t.jpg"
                subprocess.run(["sips", "-s", "format", "jpeg", "-Z", str(THUMB_W), str(path), "--out", str(out)], capture_output=True, timeout=120)
                if out.exists():
                    return out.read_bytes()
        return None

    def _probe(self, path: Path) -> dict:
        from .glasses import _display_dims
        from .localproj import timecode_df, timecode_s
        from .polish import _fps
        info = ffmpeg.probe(path)
        vs = ffmpeg.video_stream(info) or {}
        w, h = _display_dims(vs)
        rate = _fps(vs) if vs else 30.0
        return {"dur": round(float(info.get("format", {}).get("duration") or 0), 3), "w": int(w), "h": int(h), "fps": float(rate),
                "tc": timecode_s(info, rate), "tc_df": timecode_df(info), "audio": ffmpeg.audio_stream(info) is not None}

    def _store(self, f: dict, meta: dict, frames: list[bytes], step: float, motion=None) -> None:
        import numpy as np
        from PIL import Image
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames]
        try:
            emb = self.embedder().images(ims).astype("float16")
        except Exception as e:  # noqa: BLE001 - not the file's fault
            raise NotTheFile(f"the picture model failed: {e}") from e
        td = self.dir / "thumbs" / f["id"]
        shutil.rmtree(td, ignore_errors=True)
        td.mkdir(parents=True, exist_ok=True)
        for k, b in enumerate(frames):
            (td / f"{k:04d}.jpg").write_bytes(b)
        (self.dir / "emb").mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "emb" / f"{f['id']}.npy", emb)
        mver = None
        if motion is not None:                                       # where the camera really moves (cameramotion.py): a bonus, never a failure
            try:
                CM.save(self.dir / "motion" / f"{f['id']}.npy", motion)
                mver = CM.VERSION
            except Exception as e:  # noqa: BLE001
                log.info(f"camera motion not kept for {f['path']}: {e}")
        old = self.load(f["id"]) or {}
        same = old.get("size") == f["size"] and old.get("mtime") == f["mtime"]
        tags = old.get("tags") if (same and old.get("described") and old.get("tags")) else self.picture_tags(emb, f.get("rel") or "")   # (the sharpened ones stay)
        self.save({**f, **meta, "step": round(step, 3), "frames": len(frames), "model": self.model, "indexed": int(time.time()), "tags": tags,
                   "colour": "converted" if self._lut(f["path"]) else "as shot",                # (log footage: seen in normal colour)
                   "motion": mver,
                   "thumb_w": ims[0].width if ims else THUMB_W, "thumb_h": ims[0].height if ims else 0,
                   "scenes": old.get("scenes") if same else None, "described": old.get("described") if same else None})

    def _analyse_motion(self, mot: list):
        """The motion array from what _grab collected, or None (no frames, no OpenCV...: the heat map then works as it did)."""
        try:
            times, gray = mot[0] if mot else ([], None)
            return CM.analyze(times, gray) if len(times) >= 3 else None
        except Exception as e:  # noqa: BLE001
            log.info(f"camera motion not worked out: {e}")
            return None

    def index_one(self, f: dict) -> bool:
        path = Path(f["path"])
        try:
            if f["kind"] == "photo":
                b = self._photo(path)
                if not b:
                    raise RuntimeError("could not read the photo")
                self._store(f, {"dur": 0.0}, [b], 0.0)
            else:
                meta = self._probe(path)
                if meta["dur"] < 0.5 or not meta["w"]:
                    raise RuntimeError("no picture in it")
                mot: list = []
                frames, step = self._frames(path, meta["dur"], motion=mot)
                if not frames:
                    raise RuntimeError("no frame could be read")
                self._store(f, meta, frames, step, motion=self._analyse_motion(mot))
            return True
        except NotTheFile:
            raise
        except Exception as e:  # noqa: BLE001 - one unreadable file is noted and skipped, never the end of the run
            if not Path(f["root"]).is_dir():
                raise NotTheFile(f"the drive went away ({f['root']})") from e
            log.warning(f"catalog: {f['path']}: {e}")
            self.save({**f, "error": str(e)[:200], "model": self.model, "frames": 0, "indexed": int(time.time())})
            return False

    def index(self, deadline: float | None = None, limit: int | None = None, progress=None) -> dict:
        found = self.scan()
        gone = self.prune(found)
        todo = self.todo(found)
        todo.sort(key=lambda f: (f["kind"] != "video", f["rel"]))          # videos first: they are what the editor uses most
        waiting = len(todo)
        if limit:
            todo = todo[:limit]
        done = failed = 0
        stopped = None
        if todo:
            try:                                                         # the picture model first: if it can't load, read nothing
                from PIL import Image
                self.embedder().images([Image.new("RGB", (32, 32))])
            except Exception as e:  # noqa: BLE001
                stopped = f"the picture model could not load: {e}" + (" (open Settings › Setup and run it again: it installs the missing part)" if isinstance(e, ImportError) else "")
                todo = []
        # Several files at once: reading a 4K camera file off a drive is mostly waiting (the drive, ffmpeg), and one at a time left the Mac
        # idle most of the time (about 8 files a minute). The picture model still takes one picture batch at a time (its own lock).
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        from . import loadgov
        if int(self.workers) > 0:                                # a number set by hand: exactly that many
            gov = loadgov.Governor("gentle", cap=int(self.workers))
            gov.fixed, gov._target, gov._why = int(self.workers), int(self.workers), "full"
        else:
            gov = loadgov.Governor(self.load_mode)
        self.governor = gov.start()

        def work(f):
            if self.stop.is_set() or (deadline and time.time() > deadline):
                return "skip", f, None
            self._wait()
            self._live("now", step="frames", id=f["id"], rel=f["rel"], media=f["kind"], k=None)
            try:
                return ("ok" if self.index_one(f) else "fail"), f, None
            except NotTheFile as e:
                return "stop", f, str(e)
            except Exception as e:  # noqa: BLE001 - (e.g. the error note itself could not be saved)
                log.warning(f"catalog: {f.get('path')}: {e}")
                return "fail", f, None

        pending, it, n, halt, exhausted = set(), iter(todo), 0, False, False
        try:
          with ThreadPoolExecutor(max(1, gov.cap), thread_name_prefix="catalog-frames") as pool:
            while True:
                stopping = self.stop.is_set() or bool(deadline and time.time() > deadline)
                want = gov.target()                              # how many may be in flight now: the Mac decides (0 = paused)
                self.status.update(workers=want, load=gov.note())
                while not halt and not exhausted and not stopping and len(pending) < want:
                    f = next(it, None)
                    if f is None:
                        exhausted = True
                        break
                    if progress:
                        progress(n, len(todo), f"Looking at {f['rel']}")
                    pending.add(pool.submit(work, f))
                if not pending:
                    if halt or exhausted or stopping:
                        break
                    self.status["stage"] = gov.note() or "Paused"       # (paused: another app needs the Mac; nothing in flight)
                    self.stop.wait(1.0)
                    continue
                finished, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    what, f, why = fut.result()
                    n += 1
                    if what in ("ok", "fail") and getattr(gov, "note_done", None):
                        gov.note_done()                                  # (how fast files finish is how the governor finds the right count)
                    if what == "stop":
                        stopped, halt = why, True
                    elif what == "ok":
                        self._live("done", step="frames", id=f["id"], rel=f["rel"], media=f["kind"], k=self._mid(self.load(f["id"])), what=[], by=None,
                                   tags=(self.load(f["id"]) or {}).get("tags") or [])
                        done += 1
                    elif what == "fail":
                        failed += 1
                    self.compile_soon(done + failed)                     # what is done so far is searchable while the rest runs
        finally:
            gov.stop()
            self.status.update(workers=0, load="")
        try:
            self.fill_tags()                                              # (files catalogued before keywords: embeddings only, a moment's work)
        except Exception as e:  # noqa: BLE001
            log.info(f"keywords for older files not made: {e}")
        self.compile()
        if stopped:
            log.error(f"catalog: stopped: {stopped}")
        return {"found": len(found), "new": done, "failed": failed, "left": waiting - done - failed, "forgotten": gone,
                **({"stopped": stopped[:300]} if stopped else {})}

    # -- pass 2: what each scene shows (the local vision model) ---------------------------------------
    def vision(self):
        if self._vision is None and self.describe_with == "account":
            self._vision = _AccountVision(self.cfg)                       # (offline it is this Mac's picture model again: LocalAI)
        if self._vision is None:
            from ..core.llm import OllamaLLM
            self._vision = OllamaLLM({"base_url": self.ollama_url, "model": self.vision_model, "vision_model": self.vision_model,
                                      "timeout_s": 240, "keep_alive": "10m"})   # (answers are capped: one that takes longer is stuck,
                                                                                    #  and Stop waits for the file in hand)
        return self._vision

    def _sheet(self, tiles: list[tuple[bytes, str]], cols: int = 4) -> bytes:
        """A contact sheet of up to 12 labelled tiles."""
        from PIL import Image, ImageDraw, ImageFont
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for b, _ in tiles]
        tw, th = 256, 144
        rows = math.ceil(len(ims) / cols)
        sheet = Image.new("RGB", (cols * tw + (cols - 1) * 4, rows * th + (rows - 1) * 4), (20, 20, 20))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        d = ImageDraw.Draw(sheet)
        for k, (im, (_, label)) in enumerate(zip(ims, tiles)):
            im = im.copy()
            im.thumbnail((tw, th))
            x, y = (k % cols) * (tw + 4), (k // cols) * (th + 4)
            sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
            d.rectangle((x, y, x + 8 + d.textlength(label, font=font), y + 26), fill=(0, 0, 0))
            d.text((x + 4, y + 2), label, font=font, fill=(255, 235, 60))
        b = io.BytesIO()
        sheet.save(b, "JPEG", quality=82)
        return b.getvalue()

    def describe_one(self, rec: dict) -> bool:
        """The scenes of one video, from its catalogued frames (12 per contact sheet, at most 4 sheets)."""
        from . import broll as BR
        n = int(rec.get("frames") or 0)
        if not n or rec["kind"] != "video":
            return False
        pick = list(range(n)) if n <= 48 else sorted({round(k * (n - 1) / 47) for k in range(48)})
        step = float(rec.get("step") or self.every)
        tiles = []
        for k in pick:
            p = self.thumb(rec["id"], k)
            if p.exists():
                t = k * step
                tiles.append((p.read_bytes(), f"{int(t // 60)}:{t % 60:04.1f}"))
        if not tiles:
            return False
        sheets = [self._sheet(tiles[k:k + 12]) for k in range(0, len(tiles), 12)]
        times = ", ".join(f"{k * step:.1f}" for k in pick)
        prompt = (f"File: {Path(rec['path']).name} in the folder {Path(rec['rel']).parent} ({rec['dur']:.1f} s long). {len(sheets)} contact "
                  f"sheet(s); frame times in seconds: {times}. Describe the scenes. Use these times.")
        try:
            res = self.vision().complete_json(BR.SYSTEM, prompt, BR.SCHEMA, images=sheets)
            scenes = BR._valid_scenes(res.data.get("scenes"), 0.0, float(rec["dur"]))
        except Exception as e:  # noqa: BLE001
            log.warning(f"catalog: describing {rec['path']} failed: {e}")
            return False
        rec = {**rec, "scenes": scenes, "described": getattr(res, "model", None) or self.vision_model}
        rec["tags"] = self._described_tags(rec, [str(x.get("what") or "") for x in scenes])
        self.save(rec)
        self._live("done", step="describe", id=rec["id"], rel=rec["rel"], media="video", k=self._mid(rec),
                   what=[str(x.get("what") or "")[:140] for x in scenes[:2]], by=self.who(rec["described"]), tags=rec["tags"])
        return True

    def describe_photos(self, recs: list[dict]) -> int:
        """Up to 12 photos per contact sheet: one sentence and a quality score each."""
        tiles = [(self.thumb(r["id"], 0).read_bytes(), str(k + 1)) for k, r in enumerate(recs) if self.thumb(r["id"], 0).exists()]
        if not tiles:
            return 0
        try:
            res = self.vision().complete_json(TILES_SYSTEM, f"{len(tiles)} photos, numbered 1 to {len(tiles)}. Describe every one.",
                                              TILES_SCHEMA, images=[self._sheet(tiles)])
        except Exception as e:  # noqa: BLE001
            log.warning(f"catalog: describing {len(tiles)} photos failed: {e}")
            return 0
        n = 0
        for t in res.data.get("tiles") or []:
            try:
                k, what, q = int(t["n"]) - 1, " ".join(str(t["what"]).split())[:200], max(1.0, min(10.0, float(t["quality"])))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= k < len(recs) and what:
                by = getattr(res, "model", None) or self.vision_model
                tags = self._described_tags(recs[k], [what])
                self.save({**recs[k], "scenes": [{"start": 0.0, "end": 0.0, "what": what, "quality": q}], "described": by, "tags": tags})
                self._live("done", step="describe", id=recs[k]["id"], rel=recs[k]["rel"], media="photo", k=0, what=[what[:140]], by=self.who(by), tags=tags)
                n += 1
        return n

    def describe(self, deadline: float | None = None, limit: int | None = None, progress=None) -> dict:
        scope = self.mounted()
        recs = [r for r in self.records() if r.get("frames") and not r.get("described") and not r.get("error")
                and any(self.under(r.get("path", ""), root) for root in scope)]
        recs.sort(key=lambda r: (r["kind"] != "video", r["rel"]))
        if limit:
            recs = recs[:limit]
        done = k = 0
        photos = [r for r in recs if r["kind"] == "photo"]
        videos = [r for r in recs if r["kind"] == "video"]
        total = len(videos) + math.ceil(len(photos) / 12)
        for r in videos:
            if self.stop.is_set() or (deadline and time.time() > deadline):
                break
            self._wait()
            if progress:
                progress(k, total, f"Describing {r['rel']}")
            self._live("now", step="describe", id=r["id"], rel=r["rel"], media="video", k=self._mid(r))
            done += self.describe_one(r)
            self.compile_soon(done)                                      # (a description makes a shot findable by its words: at once)
            k += 1
        for s in range(0, len(photos), 12):
            if self.stop.is_set() or (deadline and time.time() > deadline):
                break
            self._wait()
            if progress:
                progress(k, total, f"Describing photos in {Path(photos[s]['rel']).parent}")
            self._live("now", step="describe", id=photos[s]["id"], rel=str(Path(photos[s]["rel"]).parent), media="photos", k=0)
            done += self.describe_photos(photos[s:s + 12])
            self.compile_soon(done)
            k += 1
        self.compile()
        return {"described": done, "left": max(0, len(recs) - done)}

    # -- searching --------------------------------------------------------------------------------
    def compile_soon(self, n: int) -> None:
        """During a run: the first files are searchable at once (after 1, 5 and 20), then at most every COMPILE_S seconds, so someone
        cataloguing a big library sees shots arrive in Find footage within a minute (it was every 300 files, and descriptions only
        after the whole pass)."""
        now = time.time()
        if n in (1, 5, 20) or now - getattr(self, "_compiled_at", 0.0) >= COMPILE_S:
            self._compiled_at = now
            try:
                self.compile()
            except Exception as e:  # noqa: BLE001 - a search index refresh never stops a run
                log.warning(f"catalog: refreshing the search index failed: {e}")

    def compile(self) -> None:
        """All embeddings in one matrix (rows of one file together), and a light copy of every record, for fast searching."""
        import numpy as np
        mats, meta, starts, n = [], [], [], 0
        for r in self.records():
            if not r.get("frames") or r.get("error"):
                continue
            p = self.dir / "emb" / f"{r['id']}.npy"
            try:
                e = np.load(p)
            except (OSError, ValueError):
                continue
            if not len(e):
                continue
            mats.append(e.astype("float16"))
            starts.append(n)
            n += len(e)
            meta.append({k: r.get(k) for k in ("id", "path", "rel", "root", "kind", "dur", "w", "h", "fps", "tc", "tc_df", "audio", "step",
                                                   "scenes", "size", "mtime", "tags")} | {"frames": len(e)})
        self.dir.mkdir(parents=True, exist_ok=True)
        emb = np.concatenate(mats) if mats else np.zeros((0, 768), "float16")
        built = time.time()                                       # both halves carry it: a search between the two renames sees a mismatch
        np.savez(self.dir / "index.new.npz", emb=emb, starts=np.array(starts, dtype="int64"), built=np.array(built))
        (self.dir / "index.new.json").write_text(json.dumps({"files": meta, "built": built}))
        (self.dir / "index.new.json").replace(self.dir / "index.json")
        (self.dir / "index.new.npz").replace(self.dir / "index.npz")
        with self._idx_lock:
            self._idx = None
        self._dirty = False

    def _index(self) -> dict | None:
        import numpy as np
        with self._idx_lock:
            p, j = self.dir / "index.npz", self.dir / "index.json"
            if not p.exists() or not j.exists():
                return None
            stamp = (p.stat().st_mtime, j.stat().st_mtime)
            if self._idx and self._idx["stamp"] == stamp:
                return self._idx
            z = np.load(p)
            doc = json.loads(j.read_text())
            meta = doc["files"]
            if ("built" in z.files and float(z["built"]) != float(doc.get("built") or 0)) or len(meta) != len(z["starts"]):
                return self._idx                                  # caught between the two renames: the last whole index until it settles
            self._idx = {"emb": z["emb"], "starts": z["starts"], "files": meta, "stamp": stamp,
                         "paths": [(m["rel"] + " " + " ".join(s.get("what", "") for s in (m.get("scenes") or [])) + " " + " ".join(m.get("tags") or [])).lower() for m in meta]}
            return self._idx

    def in_sources(self, idx) -> list[bool]:
        """Which catalogued files are under the current sources (cached until the sources or the index change)."""
        key = (tuple(str(r) for r in self.roots), idx.get("stamp"), len(idx["files"]))
        if idx.get("in_key") != key:
            idx["in_mask"] = [any(self.under(f.get("path", ""), r) for r in self.roots) for f in idx["files"]]
            idx["in_key"] = key
        return idx["in_mask"]

    def stats(self) -> dict:
        idx = self._index()
        files = [f for f, ok in zip(idx["files"], self.in_sources(idx)) if ok] if idx else []
        def count(root, kind):
            return sum(1 for f in files if f["kind"] == kind and self.under(f.get("path", ""), root))
        now = [str(r) for r in (self.status.get("scope") or [])]
        return {"roots": [{"path": str(r), "mounted": r.is_dir(), "videos": count(r, "video"), "photos": count(r, "photo"),
                           "cataloguing": bool(self.status.get("running")) and (not now or str(r) in now)} for r in self.roots],
                "videos": sum(1 for f in files if f["kind"] == "video"), "photos": sum(1 for f in files if f["kind"] == "photo"),
                "described": sum(1 for f in files if f.get("scenes")), "frames": sum(int(f.get("frames") or 0) for f in files),
                "hours": round(sum(float(f.get("dur") or 0) for f in files) / 3600, 1), "status": dict(self.status)}

    def search(self, text: str, n: int = 40, kind: str | None = None, under: str | None = None, exclude: set[str] | None = None,
               per_file: int = 1) -> list[dict]:
        """Shots that look like the words, best first: [{id, path, rel, kind, t, start, end, score, chance, k (frame), what, quality, ...}].
        The picture match decides; a word of the query in the file's folders or in its description adds a little (it's often the
        category: 'graduation', 'volleyball')."""
        import numpy as np
        idx = self._index()
        text = " ".join((text or "").split())[:300]
        if not idx or not text or not len(idx["emb"]):
            return []
        q = self.embedder().text([text])[0].astype("float32")
        E = idx["emb"]
        S = np.empty(len(E), dtype="float32")
        for a in range(0, len(E), 16384):
            S[a:a + 16384] = E[a:a + 16384].astype("float32") @ q
        starts = idx["starts"]
        best = np.maximum.reduceat(S, starts) if len(starts) else np.zeros(0)
        qw = words_of(text)
        boost = np.zeros(len(starts), dtype="float32")
        if qw:
            for i, hay in enumerate(idx["paths"]):
                hits = sum(1 for w in qw if w in hay)
                if hits:
                    boost[i] = min(0.03, 0.012 * hits)
        files = idx["files"]
        ok = np.array(self.in_sources(idx), dtype=bool)
        if kind in ("video", "photo"):
            ok &= np.array([f["kind"] == kind for f in files])
        if under:
            if str(under).startswith("/"):                           # a source folder (the page's "Look in")
                ok &= np.array([self.under(f.get("path", ""), under) for f in files])
            else:
                u = under.lower()
                ok &= np.array([u in f["rel"].lower() for f in files])
        if exclude:
            ok &= np.array([f["id"] not in exclude for f in files])
        rank = best + boost
        rank[~ok] = -9
        out = []
        emb = self.embedder()
        for i in np.argsort(-rank)[:n * 3]:
            if rank[i] <= -9 or len(out) >= n:
                break
            f = files[int(i)]
            a = int(starts[i])
            row = S[a:a + f["frames"]]
            for k in [int(x) for x in np.argsort(-row)[:per_file]]:
                out.append(self._shot(f, row, k, float(rank[i]), emb))
        out.sort(key=lambda s: -s["score"])
        out = out[:n]
        self._tags_for_shots(out, idx)
        return out

    def _tags_for_shots(self, shots: list[dict], idx: dict) -> None:
        """Keywords for the files of these results that were catalogued before keywords existed (from the embeddings the index already holds,
        and kept on their records): the page shows them under each shot."""
        need = [s for s in shots if s.get("tags") is None]
        if not need:
            return
        pos = idx.get("pos")
        if pos is None:
            pos = idx["pos"] = {f["id"]: j for j, f in enumerate(idx["files"])}
        done: dict[str, list[str]] = {}
        for s in need:
            if s["id"] not in done:
                j = pos.get(s["id"])
                tags: list[str] = []
                if j is not None:
                    a = int(idx["starts"][j])
                    frames = idx["emb"][a:a + int(idx["files"][j].get("frames") or 0)].astype("float32")
                    tags = self.picture_tags(frames, s.get("rel") or "")
                    rec = self.load(s["id"])
                    if rec and rec.get("tags") is None and self.tagger() is not None:
                        self.save({**rec, "tags": tags})
                done[s["id"]] = tags
            s["tags"] = done[s["id"]]

    def _shot(self, f: dict, row, k: int, rank: float, emb) -> dict:
        """The stretch around frame k that matches about as well: at least SHOT_MIN s where the clip allows, at most SHOT_MAX."""
        step = float(f.get("step") or self.every)
        dur = float(f.get("dur") or 0)
        a = b = k
        top = float(row[k])
        while a > 0 and float(row[a - 1]) >= top - 0.012 and (k - a + 1) * step < SHOT_MAX:
            a -= 1
        while b + 1 < len(row) and float(row[b + 1]) >= top - 0.012 and (b - a + 1) * step < SHOT_MAX:
            b += 1
        t = k * step
        start, end = max(0.0, a * step - 0.5), min(dur, b * step + min(step, 2.5)) if dur else 0.0
        if f["kind"] == "video" and end - start < SHOT_MIN:
            pad = (SHOT_MIN - (end - start)) / 2
            start, end = max(0.0, start - pad), min(dur, end + pad)
            if end - start < SHOT_MIN:
                start, end = max(0.0, end - SHOT_MIN), min(dur, max(end, start + SHOT_MIN))
        sc = next((s for s in (f.get("scenes") or []) if s["start"] - 0.5 <= t <= s["end"] + 0.5), None) if f["kind"] == "video" else \
            ((f.get("scenes") or [None])[0])
        return {"id": f["id"], "path": f["path"], "rel": f["rel"], "kind": f["kind"], "k": k, "t": round(t, 2),
                "start": round(start, 2), "end": round(end, 2), "dur": dur, "score": round(rank, 4), "chance": round(emb.chance(top), 4),
                "step": step, "frames": int(f.get("frames") or len(row)),           # (the page scrubs through the file's stills: frame k at k * step)
                "what": (sc or {}).get("what"), "quality": (sc or {}).get("quality"), "w": f.get("w"), "h": f.get("h"), "fps": f.get("fps"),
                "tc": f.get("tc", 0.0), "tc_df": f.get("tc_df", False), "audio": f.get("audio"), "size": f.get("size"), "mtime": f.get("mtime"),
                "tags": f.get("tags")}

    # -- where the good part of a video is (the player's heat map) -------------------------------------------
    def look(self, i: str) -> list[float]:
        """How good each still of a video looks, 0-1, worked out once from the stills and kept: in focus (detail), well exposed, and
        something happening (change from the still before)."""
        import numpy as np
        from PIL import Image
        rec = self.load(i)
        if not rec or rec.get("kind") != "video" or not rec.get("frames"):
            return []
        if isinstance(rec.get("look"), list) and len(rec["look"]) == int(rec["frames"]):
            return rec["look"]
        sharp, expo, change, prev = [], [], [], None
        for k in range(int(rec["frames"])):
            try:
                g = np.asarray(Image.open(self.thumb(i, k)).convert("L").resize((192, 108)), dtype="float32") / 255
            except OSError:
                g = np.zeros((108, 192), dtype="float32")
            lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
            sharp.append(float(lap.var()))
            m = float(g.mean())
            expo.append(max(0.0, 1.0 - abs(m - 0.45) / 0.45))
            change.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
            prev = g
        top = max(np.percentile(sharp, 90), 1e-6)
        ch = max(np.percentile(change, 90), 1e-6)
        out = [round(0.6 * min(1.0, a / top) + 0.25 * e + 0.15 * min(1.0, c / ch), 3) for a, e, c in zip(sharp, expo, change)]
        self.save({**rec, "look": out})
        return out

    def _camera(self, i: str, rec: dict):
        """(the camera-motion profile of a video or None, still being worked out?). Files catalogued since motion was kept have it; older
        ones get it in the background (one keyframe pass, run at low priority) and the page asks again a few seconds later."""
        path = self.dir / "motion" / f"{i}.npy"
        have = rec.get("motion", "absent")
        if have == CM.VERSION and path.exists():
            m = CM.load(path)
            return (CM.profile(m) if m is not None else None), False
        if have is None or (have != "absent" and not isinstance(have, int)):
            return None, False                                       # (it was tried when the file was indexed: nothing to find)
        jobs = self.__dict__.setdefault("_motion_jobs", set())
        with self._idx_lock:
            if i in jobs:
                return None, True
            jobs.add(i)

        def run():
            try:
                with self.__dict__.setdefault("_motion_lock", threading.Lock()):          # one at a time
                    m = CM.extract(ffmpeg.bin_path("ffmpeg"), rec["path"], BACKGROUND, timeout=max(300, float(rec.get("dur") or 60) * 4))
                CM.save(path, m)
                cur = self.load(i) or rec
                self.save({**cur, "motion": CM.VERSION if len(m) else None})
            except Exception as e:  # noqa: BLE001 - the heat map works without it
                log.info(f"camera motion for {rec.get('rel') or i}: {e}")
                cur = self.load(i) or rec
                self.save({**cur, "motion": None})
            finally:
                with self._idx_lock:
                    jobs.discard(i)
        threading.Thread(target=run, name="camera-motion", daemon=True).start()
        return None, True

    def heat(self, i: str, text: str = "") -> dict:
        """Where the good part of one video is, still by still (0-1): how well each still matches the search words (when there are any),
        how good it looks (look()), and the scene ratings from the descriptions, and where the CAMERA makes a smooth move (a push-in, a pan:
        cameramotion.py; shaky settling is pushed down). "peak" is the best still; "peak_t" (when the shot has a move) is the second the
        move starts. "motion": "pending" while an older file's motion is worked out."""
        import numpy as np
        rec = self.load(i)
        if not rec or rec.get("kind") != "video" or not rec.get("frames"):
            return {"step": self.every, "heat": [], "peak": 0}
        n, step = int(rec["frames"]), float(rec.get("step") or self.every)
        look = np.array(self.look(i) or [0.5] * n, dtype="float32")
        qual = np.full(n, 0.5, dtype="float32")
        for sc in rec.get("scenes") or []:
            a, b = int(max(0, sc.get("start", 0)) // step), int(min(n - 1, sc.get("end", 0) // step))
            if sc.get("quality") is not None and b >= a:
                qual[a:b + 1] = max(0.0, min(1.0, float(sc["quality"]) / 10))
        heat = 0.7 * look + 0.3 * qual
        text = " ".join((text or "").split())[:300]
        if text:
            try:
                E = np.load(self.dir / "emb" / f"{i}.npy").astype("float32")
                sims = E @ self.embedder().text([text])[0].astype("float32")
                lo, hi = float(np.median(sims)) - 0.01, float(sims.max())
                match = np.clip((sims - lo) / max(hi - lo, 1e-4), 0, 1)
                if len(match) == n:
                    heat = 0.6 * match + 0.4 * heat
            except Exception as e:  # noqa: BLE001 - the picture side still shows
                log.info(f"heat without the words for {i}: {e}")
        prof, pending = self._camera(i, rec)
        move_span = None
        if prof is not None:
            centres = np.arange(n) * step
            move_c = np.interp(centres, prof["t"], prof["move"])
            shake_c = np.interp(centres, prof["t"], prof["shake"])
            cam_c = np.interp(centres, prof["t"], prof["cam"])
            move_span = CM.best_move(prof)
            if move_span is not None:                                # the move dominates (less when the search words point somewhere else)
                w = 0.5 if text else 0.7
                heat = np.clip((1 - w) * heat + w * move_c - 0.3 * shake_c, 0.0, 1.0)
            else:                                                    # still or handheld: unsettled stretches fall, a locked-off camera stays as it was
                heat = np.clip(heat * (0.55 + 0.9 * cam_c), 0.0, 1.0)
        if n >= 3:
            heat = np.convolve(np.pad(heat, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        peak = int(np.argmax(heat)) if n else 0
        out = {"step": step, "heat": [round(float(x), 3) for x in heat], "peak": peak}
        if move_span is not None and n:
            a = int(max(0, min(n - 1, round(move_span[0] / step)))); b = int(max(a, min(n - 1, round(move_span[1] / step))))
            if float(heat[a:b + 1].max()) >= 0.85 * float(heat.max()):         # (the search words can still point at another moment)
                out["peak"], out["peak_t"] = a, round(float(move_span[0]), 2)
        if pending:
            out["motion"] = "pending"
        return out

    def preview(self, i: str, start: float, end: float) -> Path | None:
        """A small playable copy of one shot (540p H.264, made once and kept): camera originals such as 10-bit 4:2:2 H.264, MXF or
        AVCHD do not play in a web page."""
        rec = self.load(i)
        if not rec or rec.get("kind") != "video" or rec.get("error"):
            return None
        a = max(0.0, min(float(start), float(rec["dur"]) - 0.5))
        e = min(float(rec["dur"]), max(a + 0.5, float(end)))
        out = self.dir / "previews" / f"{i}_{a:.1f}_{e:.1f}.mp4"
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._preview_lock(out.name):                          # (clicking again while it is being made waits for that copy, not a second one)
            if out.exists():
                return out
            tmp = out.with_suffix(".part.mp4")
            r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{a:.2f}", "-i", rec["path"], "-t", f"{e - a:.2f}",
                                "-vf", "scale=-2:540" + self._lut_after(rec["path"]) + ",format=yuv420p", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-movflags", "+faststart", str(tmp)],
                               capture_output=True, timeout=600)
            if r.returncode or not tmp.exists():
                tmp.unlink(missing_ok=True)
                return None
            tmp.replace(out)
        return out

    def _preview_lock(self, name: str):
        with self._idx_lock:
            locks = self.__dict__.setdefault("_pv_locks", {})
            if len(locks) > 200:
                locks.clear()
            return locks.setdefault(name, threading.Lock())

    # -- B-roll for an interview ------------------------------------------------------------------
    def shots_for(self, ai, sentences, brief: str = "", limit: int = 48, progress=None) -> list[dict]:
        """Candidate B-roll from the whole library for an interview: the AI reads the transcript in passages and writes what footage would
        show each one; each description is searched; the best few shots of different files are kept (a scene description for each, from the
        catalog or - when a file was not described yet - by the local vision model now)."""
        passages, cur = [], []
        for s in sentences:
            cur.append(s)
            if s.end - cur[0].start >= 18.0:
                passages.append(cur)
                cur = []
        if cur:
            passages.append(cur)
        if not passages or not self._index():
            return []
        lines = [f"P{n + 1} [{p[0].start:.0f}-{p[-1].end:.0f} s] " + " ".join(s.text for s in p) for n, p in enumerate(passages)]
        prompt = ((f"The creator's notes: {brief[:1500]}\n\n" if brief else "") + "The interview in passages:\n" + "\n".join(lines)[:24000]
                  + "\n\nFor each passage (by its number), write what footage would show it.")
        try:
            res = ai.complete_json("broll_queries", QUERY_SYSTEM, prompt, QUERY_SCHEMA)
        except Exception as e:  # noqa: BLE001 - no library B-roll is a fine result
            log.warning(f"catalog: writing B-roll searches failed: {e}")
            return []
        looks = []
        for p in res.data.get("passages") or []:
            for lk in (p.get("looks") or [])[:2]:
                lk = " ".join(str(lk).split())[:120]
                if lk and lk.lower() not in {x.lower() for x in looks}:
                    looks.append(lk)
        out: list[dict] = []
        taken: set[str] = set()
        per = max(2, min(4, math.ceil(limit / max(1, len(looks))) + 1))
        try:
            for n, lk in enumerate(looks[:60]):
                if progress:
                    progress(n / max(1, len(looks)), f"Searching the library: {lk}")
                for s in self.search(lk, n=per, exclude=taken):
                    if s["score"] < self.min_score or (s["quality"] is not None and s["quality"] < 4):
                        continue
                    if s["kind"] == "photo" and Path(s["path"]).suffix.lower() not in EDIT_PHOTO_EXT:
                        continue
                    taken.add(s["id"])
                    out.append({**s, "query": lk})
                if len(out) >= limit:
                    break
        except Exception as e:  # noqa: BLE001 - library B-roll is a bonus (the picture model out of memory, say): keep what was found
            log.warning(f"catalog: searching the library failed after {len(out)} shot(s): {e}")
        try:
            self.fill_descriptions(out)
        except Exception as e:  # noqa: BLE001
            log.warning(f"catalog: describing found shots failed: {e}")
            for s in out:
                s["what"] = s.get("what") or f"(found searching for '{s.get('query') or 'footage'}') {Path(s['rel']).parent.name}"
                s["quality"] = s.get("quality") or 6.0
        return out[:limit]

    def fill_descriptions(self, shots: list[dict]) -> None:
        """A one-line description for every shot that has none yet: up to 12 shots per contact sheet (each tile its middle frame)."""
        need = [s for s in shots if not s.get("what")]
        for a in range(0, len(need), 12):
            batch = need[a:a + 12]
            tiles = [(self.thumb(s["id"], s["k"]).read_bytes(), str(n + 1)) for n, s in enumerate(batch) if self.thumb(s["id"], s["k"]).exists()]
            data = {}
            if len(tiles) == len(batch):
                try:
                    data = self.vision().complete_json(TILES_SYSTEM, f"{len(tiles)} shots, numbered 1 to {len(tiles)}. Describe every one.",
                                                       TILES_SCHEMA, images=[self._sheet(tiles)]).data
                except Exception as e:  # noqa: BLE001 - the search words stand in for the description
                    log.warning(f"catalog: describing {len(tiles)} found shots failed: {e}")
            got = {int(t.get("n", 0)) - 1: t for t in data.get("tiles") or [] if isinstance(t, dict)}
            for n, s in enumerate(batch):
                t = got.get(n)
                if t and str(t.get("what", "")).strip():
                    s["what"] = " ".join(str(t["what"]).split())[:200]
                    s["quality"] = max(1.0, min(10.0, float(t.get("quality") or 6)))
                else:
                    s["what"] = f"(found searching for '{s.get('query') or 'footage'}') {Path(s['rel']).parent.name}"
                    s["quality"] = s.get("quality") or 6.0

    # -- Final Cut ----------------------------------------------------------------------------------
    def fcpxml(self, shots: list[dict], name: str = "Found footage") -> str:
        """A Final Cut event holding the found video shots as trimmed clips (File > Import > XML). The media is referenced where it is."""
        from fractions import Fraction
        from xml.sax.saxutils import escape, quoteattr

        from .fcpxml import VERSION, Clock, file_url, std_rate
        vids = [s for s in shots if s["kind"] == "video" and s.get("end", 0) > s.get("start", 0)]
        fmts: dict[tuple, str] = {}
        assets: dict[str, tuple[str, dict]] = {}
        res, clips = [], []
        for s in vids:
            rate = std_rate(s.get("fps") or 30.0)
            key = (int(s.get("w") or 1920), int(s.get("h") or 1080), rate)
            if key not in fmts:
                fmts[key] = f"r{len(fmts) + 1}"
                res.append(f'    <format id="{fmts[key]}" frameDuration="{Clock(rate).at(1)}" width="{key[0]}" height="{key[1]}"/>')
            if s["id"] not in assets:
                aid = f"a{len(assets) + 1}"
                assets[s["id"]] = (aid, s)
                clk = Clock(rate)
                res.append(f'    <asset id="{aid}" name={quoteattr(Path(s["path"]).name)} start="{clk.t(float(s.get("tc") or 0))}" '
                           f'duration="{clk.t(float(s["dur"]))}" hasVideo="1" format="{fmts[key]}" videoSources="1"'
                           + (' hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000"' if s.get("audio") else "") + ">")
                res.append(f'      <media-rep kind="original-media" src={quoteattr(file_url(s["path"]))}/>')
                res.append("    </asset>")
        for n, s in enumerate(vids):
            aid, _ = assets[s["id"]]
            rate = std_rate(s.get("fps") or 30.0)
            clk = Clock(rate)
            label = (s.get("what") or Path(s["path"]).stem)[:80]
            clips.append(f'      <asset-clip ref="{aid}" name={quoteattr(f"{n + 1:02d} {label}")} start="{clk.t(float(s.get("tc") or 0) + s["start"])}" '
                         f'duration="{clk.t(s["end"] - s["start"])}" format="{fmts[(int(s.get("w") or 1920), int(s.get("h") or 1080), rate)]}" '
                         f'tcFormat="{"DF" if s.get("tc_df") else "NDF"}">')
            clips.append(f'        <note>{escape(s["rel"])}</note>')
            clips.append("      </asset-clip>")
        return "\n".join(['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE fcpxml>", f'<fcpxml version="{VERSION}">', "  <resources>", *res,
                          "  </resources>", "  <library>", f"    <event name={quoteattr(name[:80])}>", *clips, "    </event>", "  </library>",
                          "</fcpxml>"]) + "\n"

    # -- running it in the background ------------------------------------------------------------------
    def start(self, hours: float | None = None, describe: bool = True, limit: int | None = None, frames: bool = True,
              only: list | None = None, rest: bool = True, night: bool = False) -> bool:
        """Catalogue in a background thread (the app's "Catalogue now", the nightly window, or `vb catalog`): at most `hours`, at most
        `limit` files per pass; frames=False only describes what is already catalogued. `only`: these folders first (a folder just
        added, or picked to search), then everything else unless rest=False. False if it is already running."""
        with self._run_lock:
            if self._starting or (self.thread and self.thread.is_alive()):
                return False
            self._starting = True                                         # (claimed: a second start() now returns False)
        try:
            return self._start(hours, describe, limit, frames, only, rest, night)
        finally:
            self._starting = False

    def _start(self, hours, describe, limit, frames, only, rest, night) -> bool:
        self.stop.clear()
        deadline = time.time() + hours * 3600 if hours else None
        first = [Path(str(f)) for f in (only or []) if str(f)]
        if not night:                                                    # asked for now: if the app quits, the next launch carries on
            self.dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.dir / PENDING, {"only": [str(f) for f in first], "rest": rest, "describe": describe, "frames": frames,
                                                   "limit": limit, "until": deadline})

        def prog(k, n, what):
            self.status.update(done=k, total=n, stage=what)

        def one_pass(scope) -> bool:
            """Frames, then descriptions, for `scope` (None = every folder). False when the run has to end here."""
            self.only = scope
            self.status.update(scope=[str(r) for r in scope] if scope else None, stage="Listing the folders", done=0, total=0)
            if frames:
                r = self.index(deadline, limit=limit, progress=prog)
                self.status.update(last_index=r)
                why = self.nothing_found()
                if why:                                                  # (said on the page: an empty run used to end without a word)
                    self.status.update(error=why, stage=why[:120])
                    log.warning(f"catalog: {why}")
                if r.get("stopped"):                                     # the drive went away or the picture model failed: say so, stop
                    self.status.update(error=r["stopped"], stage="Stopped: " + r["stopped"][:120])
                    return False
            if describe and not self.stop.is_set() and not (deadline and time.time() > deadline):
                self.status.update(stage="Describing scenes", done=0, total=0)
                self.status.update(last_describe=self.describe(deadline, limit=limit, progress=prog))
            return not self.stop.is_set() and not (deadline and time.time() > deadline)

        def run():
            self.status.update(running=True, error=None, started=time.time(), done=0, total=0)
            try:
                if (not first or one_pass(first)) and not limit and (rest or not first):
                    one_pass(None)
                self.status.update(stage="Stopped" if self.stop.is_set() else "Up to date" if not (deadline and time.time() > deadline) else "Paused until tonight")
            except Exception as e:  # noqa: BLE001
                log.exception("catalog run failed")
                self.status.update(error=str(e)[:300], stage="Failed")
            finally:                                                     # (ended here, stopped or done: nothing to carry on with;
                self.only = None                                         #  a quit app never reaches this line, so it is resumed)
                (self.dir / PENDING).unlink(missing_ok=True)
                self.status.update(running=False, finished=time.time(), scope=None)
        self.thread = threading.Thread(target=run, name="catalog", daemon=True)
        self.thread.start()
        return True

    def first(self, folders: list, hours: float | None = None) -> None:
        """Catalogue these folders now, before anything else: a run already going is stopped (what it did stays done) and starts again
        with these first, plus any folder an interrupted "now" run hadn't finished. A night run carries on with the rest until the
        night window ends; in the daytime it ends with these folders. Requests that arrive together are run together, one run."""
        with self._run_lock:
            self._wanted = list(dict.fromkeys([*(str(f) for f in folders), *self._wanted]))
            if self._handing:
                return
            self._handing = True

        def go():
            try:
                if self.thread and self.thread.is_alive():
                    left = [str(x) for x in (self.status.get("scope") or [])]   # a "now" run that is cut short: its folder comes along
                    self.stop.set()
                    self.thread.join(timeout=600)                        # (it stops between two files)
                    with self._run_lock:
                        self._wanted = list(dict.fromkeys([*self._wanted, *left]))
                with self._run_lock:
                    want, self._wanted, self._handing = self._wanted, [], False
                night = bool(self.window) and in_window(self.window)
                h = hours
                if night and h is None and self.window != "always":
                    from datetime import timedelta
                    hh, mm = (int(x) for x in str(self.window[1]).split(":"))
                    now = datetime.now()
                    end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if end <= now:
                        end += timedelta(days=1)
                    h = (end - now).total_seconds() / 3600
                if want and not self.start(hours=h, only=want, rest=night):
                    with self._run_lock:                                 # (something else started in between: try again shortly)
                        self._wanted = list(dict.fromkeys([*want, *self._wanted]))
                    threading.Timer(2.0, lambda: self.first([], hours)).start()
            except Exception:  # noqa: BLE001
                log.exception("catalogue hand-over failed")
                with self._run_lock:
                    self._handing = False
        threading.Thread(target=go, name="catalog-first", daemon=True).start()

    def tick(self, busy: bool = False) -> None:
        """Called every few minutes by the app: a run asked for before the app quit carries on; inside the night window (and while no
        video is being made), keep cataloguing."""
        if not self.enabled() or busy or (self.thread and self.thread.is_alive()):
            return
        try:
            pend = json.loads((self.dir / PENDING).read_text())
        except (OSError, ValueError):
            pend = None
        if pend:
            until = pend.get("until")
            if until and time.time() > float(until):
                (self.dir / PENDING).unlink(missing_ok=True)
            elif any(Path(f).is_dir() for f in pend.get("only") or []) or (not pend.get("only") and self.mounted()):
                log.info("carrying on with the catalogue run from before the restart")
                self.start(hours=(float(until) - time.time()) / 3600 if until else None, describe=bool(pend.get("describe", True)),
                           limit=pend.get("limit"), frames=bool(pend.get("frames", True)), only=pend.get("only") or None,
                           rest=bool(pend.get("rest", True)))
                return
        if not self.window or not self.mounted():
            return
        if self.window == "always":                                      # whenever the app is open: on until everything is done, and
            if time.time() - float(self.status.get("finished") or 0) > 600:   # a look for new footage every ten minutes after that
                self.start(hours=None, night=True)
            return
        if in_window(self.window) and time.time() - float(self.status.get("finished") or 0) > 1800:
            from datetime import timedelta
            hh, mm = (int(x) for x in str(self.window[1]).split(":"))
            now = datetime.now()
            end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if end <= now:
                end += timedelta(days=1)
            self.start(hours=(end - now).total_seconds() / 3600, night=True)
