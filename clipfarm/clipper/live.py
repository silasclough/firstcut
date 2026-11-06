"""What a video being made is doing right now, for the page's live view ("watch it work").

North star (CLAUDE.md UI rules): while something is being made, its page shows the real work as it happens - the words as they are
transcribed, the frames being looked at and what was seen in them, the moments found, the lines and B-roll chosen, each clip as it is
finished - so the wait is interesting and you can see it is going the right way. Nothing here is invented: every event is something
the pipeline really did (the page types out a file's words as they arrive; it never makes words up).

Events per job: the last RING in memory (the pipeline and the server are one process), saved to <job>/live.json every few seconds so
the log is still there after a restart. Frames are small JPEGs in the job folder (<job>/live_<n>.jpg, served at /media/<job>/...).
Emitting never raises and never waits on anything slow: a live view must never slow down or break a render.

    emit(store, job, kind, **data) -> n        any event (0 when it could not be recorded)
    stage(store, job, text, status)            the activity log; a repeated "Downloading video (43%)" updates its own line
    words(store, job, words, of="")            transcribed words, in chunks
    frame(store, job, image | bytes, path=, t=) -> "live_<n>.jpg" ("" when it failed)
    filmstrip(store, job, path, dur, of="")    a few frames of a video, one event each, made in the background
    since(store, job, after) -> {"events": [...], "last": n}
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.live")

RING = 400                       # events kept per job
MAX_FRAMES = 160                 # frame files kept per job (the oldest are deleted)
MAX_WORDS = 4000                 # words sent per transcription (a long file's start is plenty to watch)
CHUNK = 60                       # words per event
JOBS = 24                        # jobs kept in memory
NAME = re.compile(r"^live_\d{1,7}\.jpg$")

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _state(store, job_id: str) -> dict:
    st = _jobs.get(job_id)
    if st is None:
        d = Path(store.dir(job_id))
        events, n, fn = [], 0, 0
        try:
            saved = json.loads((d / "live.json").read_text())
            events, n, fn = list(saved.get("events") or [])[-RING:], int(saved.get("n") or 0), int(saved.get("fn") or 0)
        except (OSError, ValueError, TypeError):
            pass
        st = {"dir": d, "events": deque(events, maxlen=RING), "n": n, "fn": fn, "saved": 0.0, "frames": deque(), "last_stage": None}
        _jobs[job_id] = st
        while len(_jobs) > JOBS:                                       # (forget the oldest job; its file stays)
            _jobs.pop(next(iter(_jobs)))
    return st


def _save(st: dict, now: bool = False) -> None:
    if not now and time.time() - st["saved"] < 2.0:
        return
    st["saved"] = time.time()
    try:
        if not st["dir"].is_dir():
            return
        tmp = st["dir"] / "live.json.tmp"
        tmp.write_text(json.dumps({"n": st["n"], "fn": st["fn"], "events": list(st["events"])}))
        tmp.replace(st["dir"] / "live.json")
    except OSError as e:
        log.debug(f"live log not saved: {e}")


def emit(store, job_id: str, kind: str, **data) -> int:
    try:
        with _lock:
            st = _state(store, job_id)
            st["n"] += 1
            st["events"].append({"n": st["n"], "t": round(time.time(), 2), "kind": kind, **data})
            _save(st, now=kind in ("clip", "done"))
            return st["n"]
    except Exception as e:  # noqa: BLE001 - never a reason for a render to fail
        log.debug(f"live event {kind} dropped: {e}")
        return 0


def shape(text: str) -> str:
    """'Downloading video (43%)' and 'Downloading video (44%)' are the same line, updated."""
    return re.sub(r"\d+([.,]\d+)?", "#", str(text or "")).strip()


def stage(store, job_id: str, text: str, status: str = "") -> int:
    """A line in the activity log. The same kind of line again (only its numbers changed) replaces the old one, at most every 0.4 s."""
    text = " ".join(str(text or "").split())[:300]
    if not text:
        return 0
    try:
        with _lock:
            st = _state(store, job_id)
            key, last = shape(text), st["last_stage"]
            if last and last["key"] == key:
                if last["text"] == text or time.time() - last["t"] < 0.4:
                    return 0
                st["events"] = deque((e for e in st["events"] if not (e["kind"] == "stage" and e.get("key") == key and e["n"] == last["n"])), maxlen=RING)
            st["n"] += 1
            ev = {"n": st["n"], "t": round(time.time(), 2), "kind": "stage", "text": text, "status": status, "key": key,
                  "replace": bool(last and last["key"] == key)}
            st["events"].append(ev)
            st["last_stage"] = ev
            _save(st)
            return st["n"]
    except Exception as e:  # noqa: BLE001
        log.debug(f"live stage dropped: {e}")
        return 0


def words(store, job_id: str, ws: list[dict], of: str = "") -> None:
    """A transcription's words for the page to type out (in chunks; at most MAX_WORDS of them), then a line saying how many."""
    try:
        ws = [w for w in (ws or []) if str(w.get("text") or "").strip()]
        if not ws:
            return
        take = ws[:MAX_WORDS]
        for i in range(0, len(take), CHUNK):
            emit(store, job_id, "words", of=of, words=[[str(w["text"]).strip(), round(float(w.get("start") or 0), 2)] for w in take[i:i + CHUNK]])
        mins = max(0.0, float(ws[-1].get("end") or ws[-1].get("start") or 0) - float(ws[0].get("start") or 0)) / 60
        emit(store, job_id, "transcribed", of=of, count=len(ws), minutes=round(mins, 1))
    except Exception as e:  # noqa: BLE001
        log.debug(f"live words dropped: {e}")


def frame(store, job_id: str, image=None, path: str | Path | None = None, t: float = 0.0, width: int = 320) -> str:
    """A small JPEG for the page: a PIL image, image bytes (JPEG/PNG), or the frame of the video at `path` at t seconds."""
    try:
        with _lock:
            st = _state(store, job_id)
            st["fn"] += 1
            name = f"live_{st['fn']}.jpg"
        out = st["dir"] / name
        if image is None and path is not None:
            from ..core import ffmpeg
            from . import logcolor                                      # (log footage seen in normal colour)
            r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{max(0.0, float(t)):.2f}", "-i", str(path), "-frames:v", "1",
                                "-vf", f"{logcolor.vf_prefix(path)}scale={int(width)}:-2", "-q:v", "6", str(out)], capture_output=True, timeout=30)
            if r.returncode or not out.exists():
                return ""
        else:
            from PIL import Image
            im = Image.open(io.BytesIO(image)) if isinstance(image, (bytes, bytearray)) else image
            im = im.convert("RGB")
            im.thumbnail((int(width), int(width) * 2))
            im.save(out, "JPEG", quality=78)
        with _lock:
            st["frames"].append(out)
            while len(st["frames"]) > MAX_FRAMES:
                st["frames"].popleft().unlink(missing_ok=True)
        return name
    except Exception as e:  # noqa: BLE001
        log.debug(f"live frame dropped: {e}")
        return ""


def later(fn, *args, **kw) -> None:
    """Run fn in the background (frames for the page are never worth waiting for)."""
    def run():
        try:
            fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            log.debug(f"live background work failed: {e}")
    threading.Thread(target=run, name="live", daemon=True).start()


def filmstrip(store, job_id: str, path: str | Path, dur: float, of: str = "", n: int = 8, what: str = "") -> None:
    """n frames spread over a video, one event each as it is made (in the background)."""
    def run():
        d = float(dur or 0)
        if d <= 0:
            return
        for k in range(n):
            t = d * (k + 0.5) / n
            f = frame(store, job_id, path=path, t=t)
            if f:
                emit(store, job_id, "frame", f=f, t=round(t, 1), of=of, what=what)
    later(run)


def shot(store, job_id: str, path: str | Path, t: float, of: str = "", what: str = "") -> None:
    """One frame of a video, as an event (call it through later())."""
    f = frame(store, job_id, path=path, t=t)
    if f:
        emit(store, job_id, "frame", f=f, t=round(float(t), 1), of=of, what=what)


def since(store, job_id: str, after: int = 0) -> dict:
    with _lock:
        st = _state(store, job_id)
        after = int(after) if 0 <= int(after) <= st["n"] else 0               # (a page from before a reset starts again)
        evs = [e for e in st["events"] if e["n"] > after]
        return {"events": evs[-250:], "last": st["n"]}


def forget(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
