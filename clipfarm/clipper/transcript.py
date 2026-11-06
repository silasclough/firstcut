"""Extract a transcript (Long form › Get the transcript): a video, a recording or a link -> its words, written down on this Mac.

North star: "Drop in a video, get its words." One action; the result is three files next to each other in the videos folder
(Transcripts/), and the words on the page with a Copy button:
    <title>.txt               plain paragraphs (a new paragraph at a long pause, or after a long run of words)
    <title> (with times).txt  the same paragraphs, each with the time it starts ([01:23])
    <title>.srt               captions: short lines, at most two seconds of reading each, for Final Cut / YouTube

Jobs are files (<home>/data/transcripts/<id>.json) and one runs at a time; one that was running when the app quit starts again
when it opens (resume()). The words come from Parakeet, the same speech-to-text the editor uses; a link is downloaded first
(audio only).

    Transcripts(root, cfg, out_dir).start(src) -> id     a path on this Mac or an http(s) link
    .list() / .get(id) / .remove(id) / .resume()
    paragraphs(words) / to_txt / to_timed / to_srt        (the formatting, tested on its own)
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.transcript")

MEDIA = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mxf", ".mts", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".aiff", ".aif"}
PAUSE_S = 1.6                      # a pause this long starts a new paragraph...
PARA_WORDS = 110                   # ...and so does a long run of words, at the next sentence end
SRT_CHARS = 42                     # a caption line
SRT_S = 3.5                        # longest a caption stays up
SPEED = 40.0                       # Parakeet writes down about this many seconds of speech per second here (for the progress bar)


# ------------------------------------------------------------------------------------------------ formatting
def _join(ws: list[dict]) -> str:
    out = ""
    for w in ws:
        t = str(w.get("text") or "").strip()
        if not t:
            continue
        out += (t if not out or re.match(r"^[,.;:!?%)\]'’]", t) or out.endswith(("(", "[", "-")) else " " + t)
    return out


def paragraphs(words: list[dict]) -> list[dict]:
    """[{"start", "end", "text"}]: a new paragraph after a long pause, or at a sentence end once a paragraph is long."""
    out, cur = [], []
    for w in words:
        if cur:
            gap = float(w["start"]) - float(cur[-1]["end"])
            ends = re.search(r"[.!?]['\"’”)]?$", str(cur[-1].get("text") or ""))
            if gap >= PAUSE_S or (len(cur) >= PARA_WORDS and ends):
                out.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]), "text": _join(cur)})
                cur = []
        cur.append(w)
    if cur:
        out.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]), "text": _join(cur)})
    return [p for p in out if p["text"]]


def clock(t: float) -> str:
    t = int(max(0.0, t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def to_txt(paras: list[dict]) -> str:
    return "\n\n".join(p["text"] for p in paras) + ("\n" if paras else "")


def to_timed(paras: list[dict]) -> str:
    return "\n\n".join(f"[{clock(p['start'])}] {p['text']}" for p in paras) + ("\n" if paras else "")


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def to_srt(words: list[dict]) -> str:
    """Captions of up to two lines of SRT_CHARS, never longer than SRT_S, broken at pauses and sentence ends."""
    caps, cur = [], []
    for w in words:
        if cur:
            text = _join(cur + [w])
            gap = float(w["start"]) - float(cur[-1]["end"])
            ended = re.search(r"[.!?]['\"’”)]?$", str(cur[-1].get("text") or ""))
            if len(text) > 2 * SRT_CHARS or float(w["end"]) - float(cur[0]["start"]) > SRT_S or gap > 0.8 or (ended and len(_join(cur)) > SRT_CHARS // 2):
                caps.append(cur)
                cur = []
        cur.append(w)
    if cur:
        caps.append(cur)
    out = []
    for n, c in enumerate(caps, 1):
        text = _join(c)
        if len(text) > SRT_CHARS:                                        # two lines, split near the middle at a space
            mid = len(text) // 2
            cut = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - mid), default=-1)
            if cut > 0:
                text = text[:cut] + "\n" + text[cut + 1:]
        end = max(float(c[-1]["end"]), float(c[0]["start"]) + 0.5)
        out.append(f"{n}\n{_srt_time(float(c[0]['start']))} --> {_srt_time(end)}\n{text}\n")
    return "\n".join(out)


def _free_name(folder: Path, stem: str, suffixes: tuple[str, ...]) -> str:
    """A name none of the files uses yet ("Talk", then "Talk 2"...)."""
    stem = re.sub(r"[/:\\\x00-\x1f]", "-", stem).strip(" .") or "Transcript"
    name, k = stem[:120], 1
    while any((folder / (name + s)).exists() for s in suffixes):
        k += 1
        name = f"{stem[:116]} {k}"
    return name


# ------------------------------------------------------------------------------------------------ jobs
class Transcripts:
    def __init__(self, root: Path, cfg, out_dir, transcribe=None, download=None):
        """root: where the job files live; out_dir: a function giving the videos folder (Settings › Where things go can change it)."""
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg, self.out_dir = cfg, out_dir
        self._transcribe = transcribe or self._parakeet
        self._download = download or self._yt_audio
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    # -- records
    def _path(self, tid: str) -> Path:
        if not re.fullmatch(r"t[0-9a-f]{10}", tid or ""):
            raise ValueError("no such transcript")
        return self.dir / f"{tid}.json"

    def get(self, tid: str) -> dict:
        p = self._path(tid)
        if not p.exists():
            raise ValueError("no such transcript")
        return json.loads(p.read_text())

    def _put(self, rec: dict) -> dict:
        rec["updated"] = time.time()
        p = self._path(rec["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(p)
        return rec

    def _update(self, tid: str, **kw) -> dict:
        with self.lock:
            return self._put({**self.get(tid), **kw})

    def list(self) -> list[dict]:
        out = []
        for p in self.dir.glob("t*.json"):
            if not re.fullmatch(r"t[0-9a-f]{10}\.json", p.name):             # (not the word lists: t….words.json)
                continue
            try:
                r = json.loads(p.read_text())
                out.append({k: r.get(k) for k in ("id", "title", "status", "stage", "progress", "created", "updated", "error", "dur", "words", "out")})
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda r: r.get("created") or 0, reverse=True)

    def remove(self, tid: str) -> None:
        """Forget it (the files it wrote stay where they are)."""
        rec = self.get(tid)
        if rec["status"] in ("queued", "working"):
            raise ValueError("it's still being written down: wait for it to finish")
        self._path(tid).unlink(missing_ok=True)
        (self.dir / f"{tid}.words.json").unlink(missing_ok=True)

    # -- running
    def start(self, src: str) -> str:
        src = str(src or "").strip()
        if re.match(r"^https?://", src):
            from . import net
            if not net.online():
                raise ValueError("links need the internet, and you're offline: choose a video on this Mac instead")
            title = re.sub(r"^https?://(www\.)?", "", src)[:80]
        else:
            p = Path(src).expanduser()
            if not p.is_file():
                raise ValueError("choose a video or a recording on this Mac, or paste a link")
            if p.suffix.lower() not in MEDIA:
                raise ValueError(f"{p.name} isn't a video or a recording")
            src, title = str(p), p.stem
        tid = "t" + uuid.uuid4().hex[:10]
        with self.lock:
            self._put({"id": tid, "src": src, "title": title, "status": "queued", "stage": "Waiting its turn", "progress": 0.0,
                       "created": time.time(), "error": None, "out": None})
        self._kick()
        return tid

    def resume(self) -> None:
        """After the app was quit: anything not finished goes again (from the start: Parakeet keeps no half-way state)."""
        for r in self.list():
            if r["status"] in ("queued", "working"):
                self._update(r["id"], status="queued", stage="Waiting its turn", progress=0.0)
        self._kick()

    def _kick(self) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._loop, name="transcripts", daemon=True)
            self.thread.start()

    def _loop(self) -> None:
        while True:
            with self.lock:                                              # (decided under the lock: a job added as the last one ends still runs)
                waiting = [r for r in self.list() if r["status"] == "queued"]
                if not waiting:
                    self.thread = None
                    return
            self._one(min(waiting, key=lambda r: r["created"] or 0)["id"])

    def _one(self, tid: str) -> None:
        rec = self._update(tid, status="working", stage="Starting", progress=0.02, error=None)
        tmp = Path(tempfile.mkdtemp(prefix="transcript-"))
        try:
            src = rec["src"]
            if re.match(r"^https?://", src):
                self._update(tid, stage="Downloading the sound", progress=0.05)
                path, title = self._download(src, tmp, lambda f: self._update(tid, progress=0.05 + 0.25 * f))
                if title:
                    self._update(tid, title=title)
            else:
                path = Path(src)
                if not path.is_file():
                    raise RuntimeError("the file isn't there any more (was a drive unplugged?)")
            dur = ffmpeg.duration(path) or 0.0
            self._update(tid, dur=round(dur, 1), stage="Writing down the words", progress=0.3)
            done = threading.Event()

            def tick():                                                      # Parakeet says nothing until it's done: estimate
                t0 = time.time()
                while not done.wait(2.0):
                    f = min(0.95, (time.time() - t0) / max(5.0, dur / SPEED))
                    try:
                        if done.is_set() or self.get(tid)["status"] != "working":
                            return
                        self._update(tid, progress=round(0.3 + 0.65 * f, 3))
                    except (OSError, ValueError):
                        return
            threading.Thread(target=tick, daemon=True).start()
            try:
                words = self._transcribe(path)
            except Exception as e:  # noqa: BLE001 - ffmpeg's words for "no sound track" are not for people
                if re.search(r"does not contain any stream|matches no streams|Output file is empty|no audio", str(e), re.I):
                    raise RuntimeError("this video has no sound track, so there are no words to write down") from None
                raise
            finally:
                done.set()
            if not words:
                raise RuntimeError("no words were heard in it")
            rec = self.get(tid)
            paras = paragraphs(words)
            folder = Path(self.out_dir()) / "Transcripts"
            folder.mkdir(parents=True, exist_ok=True)
            sfx = (".txt", " (with times).txt", ".srt")
            name = _free_name(folder, rec["title"], sfx)
            files = {"txt": folder / f"{name}.txt", "timed": folder / f"{name} (with times).txt", "srt": folder / f"{name}.srt"}
            files["txt"].write_text(to_txt(paras))
            files["timed"].write_text(to_timed(paras))
            files["srt"].write_text(to_srt(words))
            (self.dir / f"{tid}.words.json").write_text(json.dumps(words))
            self._update(tid, status="done", stage="Done", progress=1.0, words=len(words), paragraphs=paras,
                         out={k: str(v) for k, v in files.items()} | {"folder": str(folder)})
            log.info(f"transcript {tid}: {len(words)} words from {rec['title']}")
        except Exception as e:  # noqa: BLE001 - shown on the page in plain words
            log.warning(f"transcript {tid} failed: {e}")
            self._update(tid, status="failed", stage="Didn't finish", error=str(e)[:300])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # -- the real tools (replaced in tests)
    def _parakeet(self, path: Path) -> list[dict]:
        from . import parakeet
        return parakeet.transcribe(path, self.cfg)

    def _yt_audio(self, url: str, d: Path, progress) -> tuple[Path, str]:
        import yt_dlp

        def hook(h):
            tot = h.get("total_bytes") or h.get("total_bytes_estimate") or 0
            if h.get("status") == "downloading" and tot:
                progress(h.get("downloaded_bytes", 0) / tot)
        opts = {"outtmpl": str(d / "source.%(ext)s"), "format": "ba/b", "quiet": True, "no_warnings": True, "noplaylist": True,
                "progress_hooks": [hook], "retries": 5}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if info and info.get("entries"):
            info = info["entries"][0]
        got = next((p for p in d.glob("source.*") if p.suffix.lower() not in (".part", ".ytdl")), None)
        if not got:
            raise RuntimeError("that link gave no sound to download")
        return got, (info or {}).get("title") or ""
