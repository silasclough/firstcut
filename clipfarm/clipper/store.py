"""Disk-backed job store for the clipper: one folder per job under data/clipper/<id>/.

    job.json         status, progress, options, source, clips (the API returns this)
    source.mp4       the downloaded / uploaded long video
    transcript.json  word-timed transcript (cached; re-runs skip transcription)
    moments.json     the chosen moments (cached)
    clip_01.mp4 ...  finished clips, with clip_01.jpg thumbnails and clip_01.srt captions
"""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
from pathlib import Path

from ..core.state import atomic_write_json

ACTIVE = ("queued", "joining", "downloading", "transcribing", "finding", "rendering")


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[int, int, dict]] = {}     # job.json path -> (mtime_ns, size, parsed)

    def dir(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or ".." in job_id:
            raise KeyError(job_id)
        return self.root / job_id

    def create(self, source: dict, options: dict, status: str = "queued") -> dict:
        """A new job; status "draft" keeps the worker off it until its source is in place (then set it to "queued")."""
        job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        d = self.dir(job_id)
        d.mkdir(parents=True)
        job = {"id": job_id, "created": time.time(), "updated": time.time(), "status": status, "stage": "Queued",
               "progress": 0.0, "title": source.get("url") or source.get("filename") or job_id,
               "source": source, "options": options, "clips": [], "error": None, "cancel": False, "info": {}}
        self._write(job)
        return job

    def get(self, job_id: str) -> dict:
        p = self.dir(job_id) / "job.json"
        if not p.exists():
            raise KeyError(job_id)
        with self._lock:
            return json.loads(p.read_text())

    def list(self) -> list[dict]:
        """All jobs, newest first. Parsed files are cached by mtime (read-only for callers), so polling is cheap."""
        out, seen = [], set()
        for p in sorted(self.root.glob("*/job.json"), reverse=True):
            try:
                st = p.stat()
                key = str(p)
                hit = self._cache.get(key)
                if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                    out.append(hit[2])
                else:
                    data = json.loads(p.read_text())
                    self._cache[key] = (st.st_mtime_ns, st.st_size, data)
                    out.append(data)
                seen.add(key)
            except (json.JSONDecodeError, OSError):
                continue
        for k in [k for k in self._cache if k not in seen]:
            self._cache.pop(k, None)
        return out

    def update(self, job_id: str, **fields) -> dict:
        with self._lock:
            job = self.get(job_id)
            job.update(fields)
            job["updated"] = time.time()
            self._write(job)
            return job

    def set_clip(self, job_id: str, clip: dict) -> dict:
        with self._lock:
            job = self.get(job_id)
            clips = [c for c in job["clips"] if c["index"] != clip["index"]] + [clip]
            job["clips"] = sorted(clips, key=lambda c: c["index"])
            job["updated"] = time.time()
            self._write(job)
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            shutil.rmtree(self.dir(job_id), ignore_errors=True)

    def next_queued(self) -> dict | None:
        queued = [j for j in self.list() if j["status"] == "queued" and not j.get("cancel")
                  and not (j.get("source") or {}).get("local")]            # local AI-video projects run on their own thread
        return min(queued, key=lambda j: j["created"]) if queued else None

    def requeue_interrupted(self, resume_local: bool = False) -> list[str]:
        """Jobs that were mid-flight when the server stopped go back in the queue (steps are cached). resume_local (the app): a long
        video that was being made goes back in the queue too, with what it was doing (LocalRunner.resume picks it up)."""
        ids = []
        for j in self.list():
            if (j.get("source") or {}).get("local"):
                if resume_local and j.get("run") and (j["status"] in ACTIVE or j["status"] == "queued") and not j.get("cancel"):
                    self.update(j["id"], status="queued", stage="Picking up where it left off")
                elif j["status"] in ACTIVE or j["status"] == "queued":
                    self.update(j["id"], status="failed", stage="Interrupted by a restart",
                                error="The app restarted while this video was being made. If the edit was already decided, press Render again "
                                      "(the same cut); otherwise Make the video - finished steps (transcripts, sync) are kept.")
                continue
            if j["status"] == "uploading" and (j.get("options") or {}).get("project"):
                self.update(j["id"], status="failed", stage="Upload paused",     # AI-video projects: keep what arrived; the uploader resumes
                            error="The upload was paused (the server restarted). Resume it and only the missing files are sent.")
                continue
            if j["status"] == "uploading":                 # the upload connection died with the old server
                self.update(j["id"], status="failed", stage="Upload interrupted",
                            error="The upload was cut off (the server restarted). Please upload the file again.")
                for p in self.dir(j["id"]).glob("source.*"):
                    p.unlink(missing_ok=True)
                shutil.rmtree(self.dir(j["id"]) / "parts", ignore_errors=True)
                continue
            if j["status"] in ACTIVE and j["status"] != "queued" and not j.get("cancel"):
                self.update(j["id"], status="queued", stage="Queued (resumed after restart)")
                ids.append(j["id"])
        return ids

    def _write(self, job: dict) -> None:
        atomic_write_json(self.dir(job["id"]) / "job.json", job)
