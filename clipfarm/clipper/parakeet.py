"""Word-timed transcripts with NVIDIA Parakeet (MLX, free, local; English). Used for AI-video projects: sharper word
times than the small Whisper model, which matters when cuts land between words.

Runs the `parakeet-mlx` command from its own venv (clipper.ai_videos.parakeet_bin, default ~/clipfarm/.venv-parakeet) so its
dependencies stay out of the main environment. The first run downloads the model (~2.4 GB).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.parakeet")
WHOLE_FILE_S = 600.0              # up to this long, a file is transcribed in one piece
DEFAULT_BIN = "~/clipfarm/.venv-parakeet/bin/parakeet-mlx"


def binary(cfg=None) -> Path | None:
    p = Path(str((cfg.get("clipper.ai_videos.parakeet_bin") if cfg else None) or DEFAULT_BIN)).expanduser()
    return p if p.exists() else None


def words_from_json(data: dict) -> list[dict]:
    """parakeet-mlx JSON (sentences -> sub-word tokens with start/end) -> [{text, start, end}] words."""
    words: list[dict] = []
    for s in data.get("sentences") or []:
        for t in s.get("tokens") or []:
            txt = str(t.get("text", ""))
            if not txt.strip():
                continue
            a, b = float(t["start"]), float(t.get("end", t["start"] + float(t.get("duration", 0))))
            if words and not txt.startswith(" "):               # a continuation piece of the previous word
                words[-1]["text"] += txt
                words[-1]["end"] = b
            else:
                words.append({"text": txt.strip(), "start": a, "end": b})
    return words


_GLUED = re.compile(r"^([A-Za-z][A-Za-z'\u2019]+)(\d[\d,.]*\d)([^\w]*)$")      # a word of 2+ letters, then 2+ digits ("was22", "it2019?")


_GLUED_SMALL = re.compile(r"^(by|of|to|at|in|on|for|from|than|about|over|under|was|is|are|and|or)(\d[\d,.]*)([^\w]*)$", re.I)     # "by8%": a short word, then even ONE digit


def split_glued_numbers(words: list[dict]) -> list[dict]:
    """Parakeet sometimes joins a number to the word before it ("was22", "it2019?"). A letters-then-digits token becomes two words, the time
    shared by their lengths. (Words with a digit inside such as "4K" or "COVID19" are left alone: they start with a capital/digit run.)"""
    out: list[dict] = []
    for w in words:
        m = _GLUED.match(str(w.get("text", ""))) or _GLUED_SMALL.match(str(w.get("text", "")))
        if not m or m.group(1).isupper():
            out.append(w)
            continue
        a, b, tail = m.group(1), m.group(2) + m.group(3), None
        total = len(a) + len(b)
        cut = round(w["start"] + (w["end"] - w["start"]) * len(a) / total, 3)
        out += [{**w, "text": a, "end": cut}, {**w, "text": b, "start": cut}]
    return out


def transcribe(src: Path, cfg=None, timeout: float = 4 * 3600) -> list[dict]:
    exe = binary(cfg)
    if exe is None:
        raise RuntimeError("parakeet-mlx isn't installed")
    with tempfile.TemporaryDirectory(prefix="parakeet-") as tmp:
        wav = Path(tmp) / "audio.wav"
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   what="audio for Parakeet")
        env = {**os.environ, "PATH": os.pathsep.join([str(Path(ffmpeg.bin_path("ffmpeg")).parent), "/opt/homebrew/bin",
                                                      "/usr/local/bin", os.environ.get("PATH", "")])}      # parakeet shells out to ffmpeg
        from . import net
        if not net.online():                                             # offline: the model on disk, no update check
            env["HF_HUB_OFFLINE"] = "1"
        dur = ffmpeg.duration(wav)

        def run(*chunking: str):
            return subprocess.run(["nice", "-n", "10", str(exe), str(wav), "--output-format", "json", "--output-dir", tmp, *chunking],
                                  capture_output=True, text=True, timeout=timeout, env=env)
        # Parakeet's default 120 s chunks (15 s overlap) lose words where the speech was cut and joined (an edited video: a quarter of the
        # words, whole sentences, were missing). A short file is transcribed WHOLE; a long one in 5-minute chunks with a wide overlap.
        out = Path(tmp) / "audio.json"
        r = run("--chunk-duration", "0") if dur <= WHOLE_FILE_S else run("--chunk-duration", "300", "--overlap-duration", "30")
        if (r.returncode or not out.exists()) and dur <= WHOLE_FILE_S:
            r = run("--chunk-duration", "300", "--overlap-duration", "30")
        if r.returncode or not out.exists():
            raise RuntimeError(f"parakeet failed: {(r.stderr or r.stdout)[-300:]}")
        words = split_glued_numbers(words_from_json(json.loads(out.read_text())))
    log.info(f"parakeet: {len(words)} words from {src.name}")
    from . import glossary
    return glossary.correct(words, glossary.terms(cfg))                # (Settings › How it edits › Names and words)


# ------------------------------------------------------------------------------------------------ live dictation
# The microphone button shows the words while they are said. The page sends the recording in pieces (MediaRecorder, one every second);
# this keeps the pieces, decodes what has arrived, and gives the NEW audio to a long-running Parakeet process (dictate_worker.py, in
# Parakeet's own environment) that answers with the words so far. Words that two passes in a row agree on are "stable" (the page types
# them into the box for good); the rest is the "tail" (shown muted, it may still change). The final text still comes from the ordinary
# whole-file pass when the person stops, so this is only ever a preview: any failure here just means no preview.
LIVE_MAX_S = 300.0                # a dictation longer than this stops showing live words (the recording itself carries on)
LIVE_IDLE_S = 60.0                # a session that hasn't heard from the page for this long is dropped
WORKER_IDLE_S = 300.0             # the model stays loaded this long after the last dictation, then the process exits
LIVE_CHUNK_MAX = 4_000_000        # bytes in one piece
LIVE_FILE_MAX = 60_000_000


def worker_python(cfg=None) -> Path | None:
    exe = binary(cfg)
    if exe is None:
        return None
    py = exe.parent / "python"
    return py if py.exists() else None


def _decode_pcm(path: Path) -> bytes:
    """16 kHz mono 16-bit audio of a recording that is still being written (the last piece may be cut short)."""
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True, timeout=60)
    return r.stdout or b""


def _words(text: str) -> list[str]:
    return text.split()


class LiveDictation:
    """One live dictation at a time. `cmd` / `decode` / `now` are for tests."""

    def __init__(self, cfg=None, work: Path | None = None, cmd: list[str] | None = None, decode=None, now=None):
        import threading
        self.cfg = cfg
        self.work = Path(work) if work else Path(tempfile.gettempdir()) / "firstcut-dictation"
        self.cmd = cmd
        self.decode = decode or _decode_pcm
        self.now = now or __import__("time").monotonic
        self._lock = threading.Lock()               # the sessions and the worker process
        self._pass = threading.Lock()               # one pass at a time: a piece that arrives meanwhile just returns the last result
        self._proc = None
        self._ready = False
        self._used = 0.0
        self.sid = None
        self._last = 0.0
        self._file: Path | None = None
        self._fed = 0                               # samples already given to the worker
        self._prev: list[str] = []
        self._stable: list[str] = []
        self._result = {"stable": "", "tail": ""}
        self._seen = 0
        self._started = False                       # (the worker has been told this dictation began)

    # -- the worker process
    def _command(self) -> list[str]:
        if self.cmd:
            return list(self.cmd)
        py = worker_python(self.cfg)
        if py is None:
            raise RuntimeError("parakeet-mlx isn't installed")
        return [str(py), str(Path(__file__).with_name("dictate_worker.py"))]

    def _spawn(self) -> None:
        import threading
        env = dict(os.environ)
        from . import net
        if not net.online():
            env["HF_HUB_OFFLINE"] = "1"                              # (the model on disk, no update check)
        self._proc = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                      env=env, bufsize=1)
        self._ready = False
        threading.Thread(target=self._reaper, args=(self._proc,), daemon=True).start()

    def _reaper(self, proc) -> None:
        """Lets the model go after a few idle minutes (it is a few GB of memory)."""
        import time
        while proc.poll() is None:
            time.sleep(15)
            with self._lock:
                if proc is not self._proc:
                    return
                if self.sid is None and self.now() - self._used > WORKER_IDLE_S:
                    self._kill()
                    return

    def _kill(self) -> None:
        p, self._proc, self._ready = self._proc, None, False
        if p is not None and p.poll() is None:
            try:
                p.stdin.write('{"op": "quit"}\n')
                p.stdin.flush()
                p.wait(3)
            except Exception:  # noqa: BLE001
                p.kill()

    def _line(self, timeout: float) -> str:
        import select
        p = self._proc
        if not select.select([p.stdout], [], [], timeout)[0]:
            self._kill()
            raise RuntimeError("the speech process took too long")
        return p.stdout.readline()

    def _rpc(self, obj: dict) -> dict:
        p = self._proc
        if p is None or p.poll() is not None:
            raise RuntimeError("the speech process isn't running")
        if not self._ready:                                          # the first line is {"ready": true}, once the model is loaded
            line = self._line(180)
            if not line or "ready" not in line:
                raise RuntimeError("the speech process didn't start")
            self._ready = True
        p.stdin.write(json.dumps(obj) + "\n")
        p.stdin.flush()
        line = self._line(30)
        if not line:
            raise RuntimeError("the speech process stopped")
        out = json.loads(line)
        if out.get("error"):
            raise RuntimeError(out["error"])
        return out

    # -- a dictation
    def start(self) -> str:
        """A new session id. Another dictation that has gone quiet is dropped; one still going is refused."""
        with self._lock:
            if self.sid is not None and self.now() - self._last < LIVE_IDLE_S:
                raise RuntimeError("another dictation is going")
            self._end_locked()
            self.sid = f"{int(self.now() * 1000)}-{os.getpid()}"
            self._last = self.now()
            self.work.mkdir(parents=True, exist_ok=True)
            self._file = self.work / f"live-{self.sid}.part"
            self._file.write_bytes(b"")
            self._fed, self._prev, self._stable, self._seen, self._started = 0, [], [], 0, False
            self._result = {"stable": "", "tail": ""}
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()                                        # (starts loading the model now, while the person begins to speak)
            return self.sid

    def push(self, sid: str, piece: bytes) -> dict:
        """The next piece of the recording -> {"stable", "tail", "capped"}; the last result when a pass is still running."""
        with self._lock:
            if sid != self.sid or self._file is None:
                raise KeyError(sid)
            if len(piece) > LIVE_CHUNK_MAX or self._file.stat().st_size + len(piece) > LIVE_FILE_MAX:
                raise ValueError("that recording is too long")
            with open(self._file, "ab") as f:
                f.write(piece)
            self._last = self._used = self.now()
        if not self._pass.acquire(blocking=False):
            return {**self._result, "capped": False}
        try:
            return self._run_pass()
        finally:
            self._pass.release()

    def _run_pass(self) -> dict:
        pcm = self.decode(self._file)
        if len(pcm) // 2 / 16000 > LIVE_MAX_S:
            return {**self._result, "capped": True}
        new = pcm[self._fed * 2:]
        if not new:
            return {**self._result, "capped": False}
        import base64
        with self._lock:
            if not self.sid:
                return {**self._result, "capped": False}
            if not self._started:
                self._rpc({"op": "start"})
                self._started = True
            out = self._rpc({"op": "feed", "pcm": base64.b64encode(new).decode()})
        self._fed = len(pcm) // 2
        cur = _words((out.get("stable") or "") + (out.get("tail") or ""))
        agree = 0
        for x, y in zip(self._prev, cur):                            # words two passes in a row agree on are settled
            if x != y:
                break
            agree += 1
        self._prev = cur
        if cur[:len(self._stable)] == self._stable:                  # (a revision of settled words is ignored: the final pass fixes it)
            self._stable = cur[:max(len(self._stable), agree)]
        self._result = {"stable": " ".join(self._stable), "tail": " ".join(cur[len(self._stable):])}
        return {**self._result, "capped": False}

    def end(self, sid: str | None = None) -> None:
        with self._lock:
            if sid is None or sid == self.sid:
                self._end_locked()

    def _end_locked(self) -> None:
        if self._file is not None:
            self._file.unlink(missing_ok=True)
        if self.sid is not None and self._started and self._proc is not None and self._proc.poll() is None:
            try:
                self._rpc({"op": "stop"})
            except Exception:  # noqa: BLE001 - the process is replaced on the next start
                self._kill()
        self.sid, self._file = None, None
        self._used = self.now()

    def close(self) -> None:
        with self._lock:
            self._end_locked()
            self._kill()
