"""Local web server for the clipper: JSON API + the single-page GUI (clipper/web/index.html).

    GET  /                                  the GUI
    GET  /api/status                        engines (LLM chain, transcriber), option lists
    GET  /api/jobs                          all jobs (newest first)
    POST /api/jobs                          {url, options...} -> job
    POST /api/upload?name=..&options=..     raw request body = the video file -> job
    POST /api/jobs/batch                    {names: [...], options...} -> one job for several files, then
    POST /api/upload?job=<id>&index=<k>&name=..   each file (raw body); the job queues when all have arrived
    GET  /api/jobs/<id>                     one job
    POST /api/jobs/<id>/start               an uploaded job waits as a draft; this sets its options and starts it
    POST /api/jobs/<id>/cancel              stop after the current step
    POST /api/jobs/<id>/retry               re-queue (cached steps reused); {"source_type": "glasses"} redoes it as glasses
    DELETE /api/jobs/<id>                   delete the job and its files
    POST /api/jobs/<id>/clips/<n>/restyle   {layout, caption_style, pace, hook_title, title?} -> re-render one clip
    POST /api/jobs/<id>/clips/<n>/reject    auto-clipper clip: don't post it (approve = post without waiting)
    GET  /longform                          the Long-form tab (ideas yes/no, video review, backlog); see longform_web.py for /api/longform and /lf
    GET  /feed                              the swipe feed (TikTok-style review); /api/feed, /api/feed/stats,
    POST /api/feed/rate|undo|seen           {job, index, verdict: like|dislike} - takes effect after a short undo window
    GET  /media/<id>/<file>                 clips, thumbnails, srt (Range requests for the video player)

Binds to 127.0.0.1 by default. One worker thread renders jobs in order (rendering is CPU-bound).
"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..core.log import get_logger
from .engine import CAPTION_STYLES, LAYOUTS, MIX, PACES, Engine, default_options
from . import aivideo
from .localai import ledger_for, own_key
from .chatgpt_plan import ChatGPTPlan, ChatGPTPlanError
from .feed import Feed
from .longform_web import LongformWeb
from .store import Store

log = get_logger("clipper.server")
WEB = Path(__file__).parent / "web"
URL_OK = re.compile(r"^https?://", re.I)
MEDIA_OK = re.compile(r"^(clip_\d{2}(\.preview)?\.(mp4|jpg|srt|txt)|fcp_project\.zip|live_\d{1,7}\.jpg|shot_[0-9a-f]{12}\.jpg)$")   # (live_: the live view's frames; shot_: B-roll thumbnails)
HEAVY_CLIP_KEYS = ("edit", "moment", "transcript")                    # editing data the GUI list doesn't need (keeps polling light)


def clean_options(raw: dict, defaults: dict) -> dict:
    o = dict(defaults)
    try:
        o["clips"] = max(0, min(30, int(raw.get("clips", o["clips"]))))
        o["min_len"] = max(8.0, min(170.0, float(raw.get("min_len", o["min_len"]))))
        o["max_len"] = max(o["min_len"] + 5, min(180.0, float(raw.get("max_len", o["max_len"]))))
    except (TypeError, ValueError):
        pass
    if raw.get("layout") in LAYOUTS:
        o["layout"] = raw["layout"]
    if raw.get("caption_style") in CAPTION_STYLES or raw.get("caption_style") == "mix":
        o["caption_style"] = raw["caption_style"]
    if raw.get("pace") in PACES:
        o["pace"] = raw["pace"]
    if raw.get("edit_style") in ("standard", "auto", "youtube", "business"):
        o["edit_style"] = raw["edit_style"]
    if "hook_title" in raw:
        o["hook_title"] = str(raw["hook_title"]).lower() in ("1", "true", "yes", "on")
    if raw.get("language"):
        o["language"] = str(raw["language"])[:8]
    if raw.get("source_type") in ("auto", "glasses", "standard"):
        o["source_type"] = raw["source_type"]
    if "prompt" in raw:                                  # what the user is looking for: the AI reads it too
        o["prompt"] = str(raw.get("prompt") or "").strip()[:800]
    if raw.get("mode") in ("clips", "polish", "long", "asis"):
        o["mode"] = raw["mode"]
    if "long_minutes" in raw:
        try:
            o["long_minutes"] = max(0.0, min(60.0, float(raw["long_minutes"] or 0)))
        except (TypeError, ValueError):
            pass
    if "music" in raw:                                   # "", "auto" (match the content) or a mood name
        m = raw.get("music")
        o["music"] = "auto" if m is True else ("" if m in (False, None, "", "off") else str(m)[:20])
    if "fcp" in raw:
        o["fcp"] = str(raw["fcp"]).lower() in ("1", "true", "yes", "on")
    return o


class Worker(threading.Thread):
    def __init__(self, engine: Engine, store: Store, feed: Feed, projects=None):
        super().__init__(daemon=True, name="clipper-worker")
        self.engine, self.store, self.feed, self.projects = engine, store, feed, projects
        self.wake = threading.Event()
        self.busy_with: str | None = None
        self.side_jobs: list = []               # (callable) restyles, run between jobs
        self.lock = threading.Lock()

    def _relay(self):
        """Pull in clips the M2's nightly auto-clipper sent (every few minutes)."""
        if time.time() - getattr(self, "_last_relay", 0) < 300:
            return
        self._last_relay = time.time()
        try:
            from ..context import Context
            from . import relay
            ctx = Context(self.engine.cfg)
            for jid in relay.ingest(ctx):
                log.info(f"ingested auto-clipper job {jid}")
            relay.apply_verdicts(ctx)
            relay.auto_approve(ctx)
            from . import library
            library.export_pending(self.engine.cfg, self.store)      # archive copies that had to wait
        except Exception:  # noqa: BLE001
            log.error(traceback.format_exc())

    def _drip(self) -> bool:
        """Daytime: one source video every ~2 hours (the night run covers the night), so downloads are spread over the
        whole day. Returns True if it processed one."""
        cfg = self.engine.cfg
        if not (self.feed.local and cfg.get("clipper.auto.enabled") and cfg.get("clipper.auto.drip", True)):
            return False
        if time.time() - getattr(self, "_last_drip", 0) < 120:
            return False
        self._last_drip = time.time()
        from ..core import sharing
        sch = cfg.get("schedule", {}) or {}
        if sharing.in_window(cfg.now(), sch.get("night_start", "17:00"), sch.get("night_end", "08:00")):
            return False
        from ..context import Context
        from . import auto
        if not auto.Pace(cfg).ok()[0]:
            return False
        res = auto.run_auto(Context(cfg), max_jobs=1, paced=True)
        return bool(res)

    def _backlog_commentary(self) -> bool:
        """Daytime: add voiced commentary to the approved clips still waiting to post, in posting order."""
        cfg = self.engine.cfg
        if not self.feed.local or not cfg.get("clipper.commentary.enabled", True) or not cfg.get("clipper.commentary.redo_approved", True):
            return False
        from ..core import sharing
        sch = cfg.get("schedule", {}) or {}
        if sharing.in_window(cfg.now(), sch.get("night_start", "17:00"), sch.get("night_end", "08:00")):
            return False
        return self.feed.commentate_backlog_one(self.engine)

    def _newsreel(self) -> bool:
        """Daytime: a news reel for each hot search topic (trends.py) that has none yet - before the topic goes cold. True if it made one."""
        cfg = self.engine.cfg
        ncfg = cfg.get("clipper.newsreel", {}) or {}
        if not (self.feed.local and ncfg.get("enabled", True)):
            return False
        from ..core import sharing
        sch = cfg.get("schedule", {}) or {}
        if sharing.in_window(cfg.now(), sch.get("night_start", "17:00"), sch.get("night_end", "08:00")):
            return False
        import json as _json
        path = cfg.path("state") / "newsreels_today.json"
        day = cfg.now().strftime("%Y-%m-%d")
        try:
            st = _json.loads(path.read_text())
        except (OSError, ValueError):
            st = {}
        if st.get("day") != day:
            st = {"day": day, "n": 0, "fails": 0}
        if st["n"] >= int(ncfg.get("per_day", 2)) or st.get("fails", 0) >= 3 or time.time() < st.get("retry_after", 0):
            return False
        from . import newsreel
        try:
            newsreel.make(cfg, self.store, self.engine)
            st["n"] += 1
        except newsreel.Skip as e:                                   # nothing hot to cover / not enough facts or footage: look again later
            log.info(f"news reel postponed: {e}")
            st["retry_after"] = time.time() + 1800
            path.write_text(_json.dumps(st))
            return False
        except Exception as e:  # noqa: BLE001
            log.warning(f"news reel failed: {e}")
            st["fails"] = st.get("fails", 0) + 1
        path.write_text(_json.dumps(st))
        return True

    def _originals(self) -> bool:
        """Daytime: make a few original template videos a day (the 'majority' of what gets posted)."""
        cfg = self.engine.cfg
        ocfg = cfg.get("clipper.originals", {}) or {}
        if not (self.feed.local and ocfg.get("enabled", True)):
            return False
        from ..core import sharing
        sch = cfg.get("schedule", {}) or {}
        if sharing.in_window(cfg.now(), sch.get("night_start", "17:00"), sch.get("night_end", "08:00")):
            return False
        import json as _json
        path = cfg.path("state") / "originals_today.json"
        day = cfg.now().strftime("%Y-%m-%d")
        try:
            st = _json.loads(path.read_text())
        except (OSError, ValueError):
            st = {}
        if st.get("day") != day:
            st = {"day": day, "n": 0, "fails": 0}
        if st["n"] >= int(ocfg.get("per_day", 6)) or st.get("fails", 0) >= 3 or time.time() < st.get("retry_after", 0):
            return False
        from . import originals
        try:                                                         # the AI picks the angle; every 4th is a "then vs now"
            originals.make(cfg, self.store, self.engine, "then_vs_now" if st["n"] % 4 == 3 else None)
            st["n"] += 1
        except originals.Waiting as e:                               # e.g. a download slot isn't free yet: not a failure
            log.info(f"original video postponed: {e}")
            st["retry_after"] = time.time() + 1800
        except Exception as e:  # noqa: BLE001
            log.warning(f"original video failed: {e}")
            st["fails"] = st.get("fails", 0) + 1
        path.write_text(_json.dumps(st))
        return True

    def _ai_archive(self) -> bool:
        """AI-video projects: copy a finished video to the big drive (one per call, low priority). True if it did one."""
        if not self.projects:
            return False
        if time.time() - getattr(self, "_last_ai_try", 0) < 300:           # a missing drive is re-checked every 5 minutes
            return False
        for pid in self.projects.pending():
            self._last_ai_try = time.time()
            res = self.projects.archive(pid)
            return res == "done"
        return False

    def _refresh_stale(self) -> bool:
        """Daytime only (the night belongs to the cartoon): re-cut unrated clips that were made before the fix that
        keeps the ends of words, best-ranked first. Returns True if it did one."""
        cfg = self.engine.cfg
        if not self.feed.local or not cfg.get("clipper.feed.recut_old", True):
            return False
        from ..core import sharing
        sch = cfg.get("schedule", {}) or {}
        if sharing.in_window(cfg.now(), sch.get("night_start", "17:00"), sch.get("night_end", "08:00")):
            return False
        nxt = self.feed.next_stale()
        if not nxt:
            return False
        jid, idx = nxt
        job = self.store.get(jid)
        c = next(c for c in job["clips"] if c["index"] == idx)
        try:
            log.info(f"re-cutting {jid}#{idx} (word ends)")
            self.engine.restyle(jid, idx, c["layout"], c["caption_style"], c["pace"], bool(c.get("hook_title", True)))
        except Exception as e:  # noqa: BLE001 - leave this one alone, carry on with the rest
            log.warning(f"re-cut of {jid}#{idx} failed: {e}")
            self.feed._prev_failed.add(self.feed._key(jid, idx))
        return True

    def _plain_cycle(self) -> None:
        """One turn of the loop without any of the Clipper's own background work: side jobs, then queued jobs, else wait."""
        with self.lock:
            side = self.side_jobs.pop(0) if self.side_jobs else None
        if side:
            try:
                side()
            except Exception:  # noqa: BLE001
                log.error(traceback.format_exc())
            return
        job = self.store.next_queued()
        if not job:
            self.wake.wait(2.0)
            self.wake.clear()
            return
        self.busy_with = job["id"]
        try:
            self.engine.process(job["id"])
        finally:
            self.busy_with = None

    def run(self):
        for jid in self.store.requeue_interrupted(resume_local=self.engine.cfg.get("clipper.app") == "studio"):
            log.info(f"resuming interrupted job {jid}")
        bg = bool(self.engine.cfg.get("clipper.background", True))         # off in the work Video Builder: it only makes the videos you ask for
        while True:
            if not bg:
                self._plain_cycle()
                continue
            self._relay()
            try:
                self.feed.sweep()                                   # likes / dislikes whose undo window has passed
                self.feed.forward()                                 # (M2) hand them to the posting server
                if time.time() - getattr(self, "_last_auto", 0) > 300:
                    self._last_auto = time.time()
                    self.feed.autopost()                            # (M2) once it has earned your trust
                    self.feed.refresh_perf()                        # (M2) learn from how posted clips performed
            except Exception:  # noqa: BLE001
                log.error(traceback.format_exc())
            with self.lock:
                side = self.side_jobs.pop(0) if self.side_jobs else None
            if side:
                try:
                    side()
                except Exception:  # noqa: BLE001
                    log.error(traceback.format_exc())
                continue
            job = self.store.next_queued()
            if not job:
                try:
                    self.feed.ensure_previews()                     # small fast-loading copies for the swipe feed
                    self.feed.maintain()                            # transcripts for older clips + re-learn old ratings with them
                    if self._drip() or self._newsreel() or self._originals() or self._refresh_stale() or self._backlog_commentary() or self._ai_archive():
                        continue
                except Exception:  # noqa: BLE001
                    log.error(traceback.format_exc())
                self.wake.wait(2.0)
                self.wake.clear()
                continue
            self.busy_with = job["id"]
            log.info(f"processing {job['id']} ({job['title']})")
            done = self.engine.process(job["id"])
            self.busy_with = None
            if self.projects and done and done.get("status") == "done" and (done.get("options") or {}).get("project"):
                try:
                    pid = next((p for p in self.projects.ids() if self.projects.get(p)["job"] == job["id"]), None)
                    if pid:
                        self.projects.finished(pid)
                except Exception:  # noqa: BLE001
                    log.error(traceback.format_exc())
            if done and done.get("status") == "done" and (done.get("options") or {}).get("mode") == "asis":
                try:
                    self.feed.queue_own(job["id"], 1)                  # your own finished video: straight to the front of the line
                except Exception:  # noqa: BLE001
                    log.error(traceback.format_exc())


LIB_ID = re.compile(r"^[0-9a-f]{16}\Z")


def make_handler(engine: Engine, store: Store, worker: Worker, feed: Feed, max_upload_mb: float, projects=None, runner=None):
    jobs_cache: dict = {"t": 0.0, "body": b"", "lock": threading.Lock()}
    chatgpt_plan = ChatGPTPlan(engine.cfg)
    ai_cache: dict = {"t": 0.0, "label": ""}
    studio_api = None
    if engine.cfg.get("clipper.app") == "studio":                     # FirstCut, the app (studio_web.py): its own page and /api/app
        from .studio_web import StudioAPI
        studio_api = StudioAPI(engine.cfg, store, worker, projects, runner, chatgpt_plan, default_options, clean_options)

    def ai_label() -> str:
        """Which AI makes the edit decisions here (for the page), looked up at most once a minute."""
        if time.time() - ai_cache["t"] > 60:
            from .localai import LocalAI
            try:
                ok, why = LocalAI(engine.cfg).available()
                ai_cache["label"] = why if ok else f"no AI: {why}"
            except Exception as e:  # noqa: BLE001
                ai_cache["label"] = f"no AI: {e}"
            ai_cache["t"] = time.time()
        return ai_cache["label"]

    def library_brief() -> dict | None:
        cat = runner.library() if runner else None
        if cat is None:
            return None
        st = cat.stats()
        return {k: st[k] for k in ("videos", "photos", "described", "hours", "roots", "status")}
    longform = LongformWeb(engine.cfg)

    def jobs_payload() -> bytes:
        """The /api/jobs response, rebuilt at most every 2 s however many pages are polling it."""
        with jobs_cache["lock"]:
            if time.time() - jobs_cache["t"] < 2.0 and jobs_cache["body"]:
                return jobs_cache["body"]
            from . import relay
            out = []
            for job in store.list():
                if (job.get("options") or {}).get("project"):
                    continue                                             # AI-video projects have their own page (/ai)
                j = {k: v for k, v in job.items() if k != "cancel"}
                j["active"] = worker.busy_with == job["id"]
                clips = job.get("clips", [])
                if (job.get("source") or {}).get("auto") and any(c.get("review") for c in clips):
                    states = relay.clip_states(None, job, feed.queue_info)
                    clips = [{**c, "post": states.get(c["index"], {})} for c in clips]
                j["clips"] = [{k: v for k, v in c.items() if k not in HEAVY_CLIP_KEYS} for c in clips]
                out.append(j)
            jobs_cache["body"], jobs_cache["t"] = json.dumps(out).encode(), time.time()
            return jobs_cache["body"]

    class Handler(BaseHTTPRequestHandler):
        server_version = "Clipper/1.0"
        protocol_version = "HTTP/1.1"                  # keep-alive: a feed of videos reuses one connection

        def log_message(self, fmt, *args):  # quieter console
            if not self.path.startswith(("/api/jobs", "/media")) or "POST" in fmt % args:
                log.debug(fmt % args)

        # -------------------------------------------------------------- helpers
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg):
            self._json({"error": msg}, code)

        def _body(self, limit: int = 1_000_000) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                raise ValueError("body too large")
            raw = self.rfile.read(n) if n else b"{}"
            b = json.loads(raw or b"{}")
            if not isinstance(b, dict):
                raise ValueError("the body must be a JSON object")
            return b

        def _public(self, job: dict) -> dict:
            j = {k: v for k, v in job.items() if k != "cancel"}
            j["active"] = worker.busy_with == job["id"]
            clips = job.get("clips", [])
            if (job.get("source") or {}).get("auto") and any(c.get("review") for c in clips):
                from . import relay
                states = relay.clip_states(None, job, feed.queue_info)
                clips = [{**c, "post": states.get(c["index"], {})} for c in clips]
            j["clips"] = [{k: v for k, v in c.items() if k not in HEAVY_CLIP_KEYS} for c in clips]
            return j

        # -------------------------------------------------------------- GET
        def _not_in_app(self, parts: list[str]) -> bool:
            """FirstCut (the app) never posts: the feed, posting and long-form channel routes don't exist there."""
            if not studio_api:
                return False
            return (parts[:1] in (["feed"], ["longform"], ["lf"]) or parts[:2] in (["api", "feed"], ["api", "longform"], ["api", "review"])
                    or (len(parts) >= 4 and parts[:2] == ["api", "jobs"] and any(w in parts[-1] for w in ("post", "approve", "reject", "feed"))))

        def do_GET(self):
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
            if self._not_in_app(parts):
                return self._err(404, "not in FirstCut")
            try:
                if not parts or parts == ["index.html"]:
                    app = engine.cfg.get("clipper.app")
                    return self._file(WEB / ("studio.html" if app == "studio" else "ai.html" if app else "index.html"))
                if parts[:3] == ["api", "app", "music"] and studio_api and len(parts) >= 4:   # the music library: search, and a song to play
                    try:
                        if parts[3] == "search":
                            q = parse_qs(u.query)
                            return self._json(studio_api.music_search((q.get("q") or [""])[0], int((q.get("n") or ["60"])[0])))
                        if parts[3] == "find":                            # Find my music downloads: the scan's progress and result (musicfind.py)
                            return self._json(studio_api.musicfind_state())
                        if parts[3] == "online":                          # Openverse / Epidemic Sound
                            q = parse_qs(u.query)
                            return self._json(studio_api.music_online((q.get("source") or ["openverse"])[0], (q.get("q") or [""])[0],
                                                                      int((q.get("page") or ["1"])[0] or 1)))
                        if parts[3] == "preview":                         # Epidemic Sound: a link to hear the song
                            q = parse_qs(u.query)
                            return self._json(studio_api.music_preview((q.get("source") or [""])[0], (q.get("id") or [""])[0]))
                        if parts[3] == "stream" and len(parts) == 5:
                            return self._file(studio_api.music_file(parts[4]))
                        if parts[3] == "wave":                            # the scrub bar's waveform
                            q = parse_qs(u.query)
                            return self._json(studio_api.music_wave((q.get("source") or [""])[0], (q.get("id") or [""])[0]))
                    except (KeyError, ValueError) as e:
                        return self._err(404, str(e))
                    except Exception as e:  # noqa: BLE001 - the online source didn't answer: say so, plainly
                        return self._err(502, f"the music service didn't answer: {str(e)[:160]}")
                if parts in (["api", "app", "probe"], ["api", "app", "thumb"], ["api", "app", "localfile"]) and studio_api:   # what was dropped or chosen
                    q = parse_qs(u.query)                                  # (thumb and localfile serve only paths that probe has seen: dropped or chosen, never any other)
                    path = (q.get("path") or [""])[0]
                    try:
                        if parts[2] == "probe":
                            return self._json(studio_api.drop_probe(path))
                        if parts[2] == "thumb":
                            try:
                                at = float((q.get("t") or ["1"])[0])
                            except ValueError:
                                at = 1.0
                            return self._file(studio_api.drop_thumb(path, at))
                        return self._file(studio_api.drop_file(path))
                    except ValueError as e:
                        return self._err(400, str(e))
                    except KeyError as e:
                        return self._err(404, str(e).strip("'\""))
                if parts == ["api", "app", "stock", "search"] and studio_api:      # stock videos and photos (Find footage › Stock)
                    q = parse_qs(u.query)
                    try:
                        return self._json(studio_api.stock_search((q.get("q") or [""])[0], (q.get("kinds") or [""])[0], int((q.get("page") or ["1"])[0] or 1)))
                    except ValueError as e:
                        return self._err(400, str(e))
                if parts[:3] == ["api", "app", "gfx"] and studio_api and len(parts) == 5:   # a graphics-style preview picture
                    try:
                        pic = studio_api.gfx_preview(parts[3], parts[4], (parse_qs(u.query).get("src") or [""])[0])
                    except (KeyError, ValueError) as e:
                        return self._err(404, str(e))
                    return self._file(pic, cache=True)
                if studio_api and len(parts) == 7 and parts[:3] == ["api", "app", "long"] and parts[4] == "shots" and parts[6] == "search" \
                        and parts[5].isdigit():                               # the Replace sheet's search box: this project + the footage library
                    try:
                        return self._json(studio_api.shot_search(parts[3], int(parts[5]), (parse_qs(u.query).get("q") or [""])[0]))
                    except ValueError as e:
                        return self._err(409, str(e))
                    except KeyError:
                        return self._err(404, "no such video")
                if studio_api and len(parts) == 7 and parts[:3] == ["api", "app", "long"] and parts[4] == "shots" and parts[6] in ("media", "preview") \
                        and parts[5].isdigit() and getattr(studio_api, "runner", None):          # the trim editor: the shot's original file, or a small copy of a range
                    try:
                        if parts[6] == "media":
                            return self._file(studio_api.runner.shot_source_path(parts[3], int(parts[5])))
                        qq = parse_qs(u.query)
                        pv = studio_api.runner.shot_preview(parts[3], int(parts[5]), float((qq.get("s") or ["0"])[0]), float((qq.get("e") or ["6"])[0]))
                        return self._file(pv, cache=True) if pv else self._err(404, "no preview for that")
                    except (ValueError, TypeError) as e:
                        return self._err(409, str(e))
                    except KeyError:
                        return self._err(404, "no such video")
                if parts[:2] == ["api", "app"] and studio_api:
                    code, obj = studio_api.get(parts[2:])
                    return self._json(obj, code)
                if parts[:2] == ["api", "library"]:
                    return self._library_safe(self._library_get, parts[2:], parse_qs(u.query))
                if parts == ["feed"]:
                    return self._file(WEB / "feed.html")
                if parts == ["ai"]:
                    return self._file(WEB / "ai.html")
                if parts == ["longform"]:
                    return self._file(WEB / "longform.html")
                if parts == ["api", "longform"]:
                    return self._json(longform.payload())
                if parts[:1] == ["lf"] and len(parts) >= 2:
                    mp = longform.media("/".join(parts[1:]))
                    return self._file(mp, cache=True) if mp else self._err(404, "not found")
                if len(parts) == 5 and parts[:3] == ["api", "ai", "projects"] and parts[4] == "uploads":
                    return self._json(projects.uploads(parts[3]))
                if len(parts) == 5 and parts[:3] == ["api", "ai", "projects"] and parts[4] == "plan":
                    pr = projects.get(parts[3])
                    jb = store.get(pr["job"])
                    cl = (jb.get("clips") or [None])[0]
                    if not cl:
                        return self._err(404, "not finished yet")
                    return self._json({"plan": projects.plan_md(pr, jb, cl, store.dir(pr["job"]))})
                if parts == ["api", "ai", "projects"]:
                    return self._json({"projects": projects.list(), "rules": projects.rules(), "dir": str(aivideo.root_dir(engine.cfg)),
                                       "drive_host": aivideo.drive_host(engine.cfg), "busy_with": worker.busy_with,
                                       "local_enabled": bool(runner and runner.enabled()), "local_running": runner.running if runner else None,
                                       "usage": ledger_for(engine.cfg).summary(), "has_own_key": bool(own_key(engine.cfg)),
                                       "chatgpt_plan": chatgpt_plan.status(), "app": engine.cfg.get("clipper.app"),
                                       "out_dir": str(Path(str(engine.cfg.get("clipper.ai_videos.out_dir") or "")).expanduser()) if engine.cfg.get("clipper.ai_videos.out_dir") else "",
                                       "ai": ai_label(), "library": library_brief()})
                if parts == ["api", "ai", "chatgpt-plan"]:
                    return self._json(chatgpt_plan.status())
                if parts == ["api", "ai", "estimate"]:
                    q = parse_qs(u.query)
                    if not (runner and runner.enabled()):
                        return self._err(403, "working on a folder of this computer is switched off here (clipper.ai_videos.local_projects)")
                    from . import localproj
                    p = Path(unquote((q.get("path") or [""])[0])).expanduser()
                    if not p.is_dir():
                        return self._err(400, f"{p} is not a folder on this computer")
                    return self._json(localproj.summarize(p, engine.cfg, transcriber=(q.get("transcriber") or ["parakeet"])[0]))
                if parts == ["api", "status"]:
                    st = engine.status()
                    st.update({"busy_with": worker.busy_with, "defaults": default_options(engine.ccfg),
                               "mix": MIX, "feed": feed.on})
                    return self._json(st)
                if parts == ["api", "feed"]:
                    q = parse_qs(u.query)
                    skip = {k for k in (q.get("skip") or [""])[0].split(",") if k}
                    items = feed.items(limit=int((q.get("limit") or ["30"])[0]), exclude=skip)
                    return self._json({"items": items, "stats": feed.stats()})
                if parts == ["api", "feed", "stats"]:
                    return self._json(feed.stats())
                if parts == ["api", "jobs"]:
                    body = jobs_payload()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return None
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    return self._json(self._public(store.get(parts[2])))
                if len(parts) == 3 and parts[0] == "media" and MEDIA_OK.match(parts[2]):
                    qs = parse_qs(u.query)
                    return self._file(store.dir(parts[1]) / parts[2], download="dl" in qs, cache="v" in qs)
                return self._err(404, "not found")
            except KeyError:
                return self._err(404, "no such job")
            except (BrokenPipeError, ConnectionResetError):
                return None

        # -------------------------------------------------------------- FirstCut (the app)
        def _studio_post(self, rest: list[str]):
            from ..core.llm import LLMUnavailable
            if rest[:2] == ["dictate", "live"] and len(rest) in (3, 4):   # words while they are said: start / a piece of the recording / end
                try:
                    if rest[2] == "start":
                        return self._json(studio_api.dictate_live("start"))
                    if len(rest) == 4 and rest[3] == "end":
                        return self._json(studio_api.dictate_live("end", rest[2]))
                    n = int(self.headers.get("Content-Length") or 0)
                    if not 0 < n <= 4_000_000:
                        return self._err(400, "a piece of a recording")
                    return self._json(studio_api.dictate_live("push", rest[2], self.rfile.read(n)))
                except KeyError:
                    return self._err(404, "that dictation is over")
                except (ValueError, RuntimeError) as e:
                    return self._err(409, str(e))
            if rest == ["dictate"]:                                       # a recording from the page's microphone button (raw audio)
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= 40_000_000:
                    return self._err(400, "say something first (up to a few minutes)")
                try:
                    return self._json(studio_api.dictate(self.rfile.read(n), self.headers.get("Content-Type") or ""))
                except (ValueError, RuntimeError) as e:
                    return self._err(400, str(e))
            try:
                big = len(rest) == 3 and rest[0] == "styles" and rest[2] == "import"     # an imported skill file (base64): up to ~20 MB
                code, obj = studio_api.post(rest, self._body(limit=28_000_000 if big else 1_000_000))
            except PermissionError as e:                                  # macOS said no (the privacy prompt was answered "Don't Allow")
                return self._err(400, f"FirstCut isn't allowed to open {Path(e.filename).name if e.filename else 'that'}: allow it in System Settings › "
                                      "Privacy & Security › Files and Folders (or Full Disk Access), then try again")
            except (ValueError, TypeError) as e:
                return self._err(400, str(e))
            except KeyError:
                return self._err(404, "not found")
            except LLMUnavailable as e:
                return self._err(503, f"the AI didn't answer: {e}")
            except Exception as e:  # noqa: BLE001 - say what went wrong instead of dropping the connection
                log.error(traceback.format_exc())
                return self._err(500, str(e)[:300])
            return self._json(obj, code)

        # -------------------------------------------------------------- the footage catalog (Find footage)
        def _library_safe(self, fn, *a):
            """Every footage-catalog request gets an answer, never a closed connection (the page can only say "Load failed" then): a bad
            number or shot is the caller's mistake (400); a picture model that can't load, or anything else, says what went wrong (500)."""
            try:
                return fn(*a)
            except (BrokenPipeError, ConnectionResetError):
                return None
            except (ValueError, TypeError, AttributeError) as e:
                return self._err(400, f"that request isn't right: {e}")
            except Exception as e:  # noqa: BLE001
                log.error(traceback.format_exc())
                return self._err(500, f"the footage catalog failed: {e}")

        def _library_get(self, rest: list[str], q: dict):
            cat = runner.library() if runner else None
            if cat is None:
                return self._err(404, "there is no footage catalog on this Mac (catalog.roots)")
            if rest == ["status"]:
                return self._json(cat.stats())
            if rest == ["search"]:
                n = max(1, min(120, int((q.get("n") or ["48"])[0] or 48)))
                shots = cat.search((q.get("q") or [""])[0], n=n, kind=(q.get("kind") or [""])[0] or None, under=(q.get("under") or [""])[0] or None)
                keep = ("id", "rel", "kind", "k", "t", "start", "end", "dur", "score", "chance", "what", "quality", "w", "h")
                return self._json({"shots": [{k: s.get(k) for k in keep} for s in shots]})
            if len(rest) == 3 and rest[0] == "thumb" and LIB_ID.match(rest[1]) and re.fullmatch(r"[0-9]{1,6}", rest[2]):
                return self._file(cat.thumb(rest[1], int(rest[2])), cache=True)
            if len(rest) == 2 and rest[0] == "heat" and LIB_ID.match(rest[1]):             # where the good part of the video is
                return self._json(cat.heat(rest[1], (q.get("q") or [""])[0]))
            if len(rest) == 2 and rest[0] == "preview" and LIB_ID.match(rest[1]):          # a playable copy of one shot
                try:
                    a, e = float((q.get("s") or ["0"])[0]), float((q.get("e") or ["6"])[0])
                except ValueError:
                    return self._err(400, "s and e are seconds")
                pv = cat.preview(rest[1], a, min(e, a + 60))
                return self._file(pv, cache=True) if pv else self._err(404, "no preview for that")
            if len(rest) == 2 and rest[0] == "media" and LIB_ID.match(rest[1]):
                rec = cat.load(rest[1])                                       # only files the catalog knows: never an arbitrary path
                if not rec or rec.get("error"):
                    return self._err(404, "not in the catalog")
                return self._file(Path(rec["path"]))
            return self._err(404, "not found")

        def _library_post(self, rest: list[str], b: dict):
            import subprocess
            cat = runner.library() if runner else None
            if cat is None:
                return self._err(404, "there is no footage catalog on this Mac (catalog.roots)")
            if rest == ["start"]:
                hours = float(b["hours"]) if b.get("hours") else None
                limit = int(b["limit"]) if b.get("limit") else None
                return self._json({"started": cat.start(hours=hours, describe=bool(b.get("describe", True)), limit=limit,
                                                        frames=bool(b.get("frames", True))), **cat.stats()})
            if rest == ["stop"]:
                cat.stop.set()
                return self._json(cat.stats())
            if rest == ["reveal"]:
                rec = cat.load(str(b.get("id") or "")) if LIB_ID.match(str(b.get("id") or "")) else None
                if not rec:
                    return self._err(404, "not in the catalog")
                subprocess.run(["open", "-R", rec["path"]], capture_output=True, timeout=30)
                return self._json({"ok": True})
            if rest == ["fcpxml"]:                                            # the shots you picked, as a Final Cut event of trimmed clips
                from . import fcpxml as FX
                from .library import safe
                if not isinstance(b.get("shots") or [], list):
                    return self._err(400, "shots must be a list of {id, start, end}")
                shots = []
                for it in (b.get("shots") or [])[:200]:
                    if not isinstance(it, dict):
                        continue
                    rec = cat.load(str(it.get("id") or "")) if LIB_ID.match(str(it.get("id") or "")) else None
                    if not rec or rec.get("kind") != "video" or rec.get("error"):
                        continue
                    a = max(0.0, float(it.get("start") or 0))
                    e = min(float(rec["dur"]), float(it.get("end") or a + 5))
                    if e - a >= 0.5:
                        shots.append({**rec, "start": a, "end": e, "what": str(it.get("what") or "")[:120] or None})
                if not shots:
                    return self._err(400, "pick at least one video shot")
                name = " ".join(str(b.get("name") or "Found footage").split())[:60] or "Found footage"
                base = Path(str(engine.cfg.get("clipper.ai_videos.out_dir") or "~/Movies")).expanduser() / "Found footage"
                base.mkdir(parents=True, exist_ok=True)
                xml = base / f"{safe(name, 50)} {time.strftime('%Y-%m-%d %H%M%S')}.fcpxml"
                xml.write_text(cat.fcpxml(shots, name))
                problems = FX.validate(xml)
                if b.get("open") and not problems:
                    subprocess.run(["open", str(xml)], capture_output=True, timeout=30)       # Final Cut opens it and imports the event
                return self._json({"path": str(xml), "clips": len(shots), "problems": problems[:5]})
            return self._err(404, "not found")

        def _file(self, path: Path, download: bool = False, cache: bool = False):
            if not path.is_file():
                return self._err(404, "not found")
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start >= size or start > end:                  # (a range that starts past the end, or backwards: not satisfiable)
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if path.suffix == ".html":
                self.send_header("Cache-Control", "no-cache")     # always load the current GUI
            elif cache:                                           # ?v=<render time>: a new render has a new URL
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.parent.name}-{path.name}"')
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        # -------------------------------------------------------------- POST / DELETE
        def do_POST(self):
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
            self.close_connection = True                    # bodies (uploads) are streamed; never reuse after a POST
            jobs_cache["t"] = 0.0
            if self._not_in_app(parts):
                return self._err(404, "not in FirstCut")
            try:
                if parts == ["api", "feed", "rate"]:
                    b = self._body()
                    return self._json(feed.rate(str(b["job"]), int(b["index"]), str(b["verdict"]),
                                                immediate=(str(b["verdict"]) == "dislike"),       # deletes are final: no undo
                                                watched=(float(b["watched"]) if b.get("watched") is not None else None),
                                                duration=(float(b["duration"]) if b.get("duration") else None)))
                if parts == ["api", "feed", "undo"]:
                    b = self._body()
                    return self._json({"undone": feed.undo(str(b["job"]), int(b["index"]))})
                if parts == ["api", "feed", "seen"]:
                    b = self._body()
                    feed.seen(str(b["job"]), int(b["index"]), float(b.get("watched", 0)), float(b.get("duration", 0)))
                    return self._json({"ok": True})
                if parts == ["api", "jobs"]:
                    b = self._body()
                    url = str(b.get("url", "")).strip()
                    if not URL_OK.match(url):
                        return self._err(400, "paste a full http(s) link")
                    job = store.create({"url": url}, clean_options(b, default_options(engine.ccfg)))
                    worker.wake.set()
                    return self._json(self._public(job), 201)
                if parts == ["api", "upload"]:
                    return self._upload(parse_qs(u.query))
                if parts[:2] == ["api", "library"]:
                    return self._library_safe(self._library_post, parts[2:], self._body())
                if parts[:2] == ["api", "app"] and studio_api:
                    return self._studio_post(parts[2:])
                if parts[:2] == ["api", "ai"]:
                    return self._ai_post(parts[2:])
                if parts[:2] == ["api", "longform"]:
                    obj, code = longform.post(parts[2:], self._body())
                    return self._json(obj, code)
                if parts == ["api", "jobs", "batch"]:
                    b = self._body()
                    names = [Path(str(n)).name for n in (b.get("names") or [])][:200]
                    if len(names) < 2:
                        return self._err(400, "a batch needs at least 2 files")
                    bad = [n for n in names if Path(n).suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v")]
                    if bad:
                        return self._err(400, f"not a video file: {bad[0]}")
                    job = store.create({"files": names}, clean_options(b, default_options(engine.ccfg)))
                    store.update(job["id"], status="uploading", stage=f"Uploading 0 of {len(names)}",
                                 title=f"{len(names)} videos: {Path(names[0]).stem} ...")
                    return self._json(self._public(store.get(job["id"])), 201)
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "start":
                    job = store.get(parts[2])
                    if job["status"] != "draft":
                        return self._err(409, "this job isn't waiting for you to start it")
                    store.update(job["id"], options=clean_options(self._body(), job["options"]), status="queued",
                                 stage="Queued", error=None)
                    worker.wake.set()
                    return self._json(self._public(store.get(job["id"])))
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in ("cancel", "retry"):
                    job = store.get(parts[2])
                    if parts[3] == "cancel":
                        store.update(job["id"], cancel=True,
                                     **({"status": "cancelled", "stage": "Cancelled"} if job["status"] == "queued" else {}))
                    else:
                        b = self._body() if int(self.headers.get("Content-Length") or 0) else {}
                        st = b.get("source_type")
                        if st in ("auto", "glasses", "standard") and st != job["options"].get("source_type"):
                            if worker.busy_with == job["id"]:
                                return self._err(409, "job is running - cancel it first")
                            d = store.dir(job["id"])                  # new source type: pick + render again
                            for f in [d / "moments.json", *d.glob("clip_*")]:
                                shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
                            store.update(job["id"], options={**job["options"], "source_type": st}, clips=[])
                        store.update(job["id"], cancel=False, status="queued", stage="Queued", error=None)
                        worker.wake.set()
                    return self._json(self._public(store.get(job["id"])))
                if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "clips" and parts[5] == "restyle":
                    return self._restyle(parts[2], int(parts[4]), self._body())
                if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "clips" and parts[5] == "edit":
                    return self._edit(parts[2], int(parts[4]), self._body())
                if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "clips" and parts[5] == "post_next":
                    return self._json({"result": feed.queue_own(parts[2], int(parts[4]))})
                if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "clips" and parts[5] in ("reject", "approve"):
                    from ..context import Context
                    from . import relay
                    ctx = Context(engine.cfg)
                    fn = relay.reject_clip if parts[5] == "reject" else relay.approve_clip
                    return self._json({"result": fn(ctx, parts[2], int(parts[4]))})
                return self._err(404, "not found")
            except KeyError:
                return self._err(404, "no such job")
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self._err(400, str(e))

        def do_DELETE(self):
            self.close_connection = True
            jobs_cache["t"] = 0.0
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
            if len(parts) == 4 and parts[:3] == ["api", "ai", "projects"]:
                try:
                    proj = projects.get(parts[3])
                except KeyError:
                    return self._err(404, "no such project")
                if worker.busy_with == proj["job"] or (runner and runner.running == parts[3]):
                    return self._err(409, "it's being made right now - stop it first")
                projects.delete(parts[3])
                return self._json({"deleted": parts[3]})
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                try:
                    job = store.get(parts[2])
                except KeyError:
                    return self._err(404, "no such job")
                if worker.busy_with == job["id"]:
                    store.update(job["id"], cancel=True)
                    return self._err(409, "job is running - cancelled; delete again once it stops")
                store.delete(job["id"])
                return self._json({"deleted": job["id"]})
            return self._err(404, "not found")

        def _receive(self, dest: Path, n: int) -> bool:
            left = n
            with dest.open("wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            return left == 0

        def _upload(self, q: dict):
            name = Path((q.get("name") or ["upload.mp4"])[0]).name
            ext = Path(name).suffix.lower()
            if ext not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
                return self._err(400, "upload an .mp4, .mov, .mkv, .webm or .m4v file")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > max_upload_mb * 1024 * 1024:
                return self._err(413, f"file must be under {max_upload_mb:.0f} MB")
            if q.get("job"):                                 # one file of a multi-file job
                job = store.get(q["job"][0])
                names = job["source"].get("files") or []
                k = int((q.get("index") or ["-1"])[0])
                if job["status"] != "uploading" or not (0 <= k < len(names)):
                    return self._err(400, "this job isn't waiting for that file")
                pd = store.dir(job["id"]) / "parts"
                pd.mkdir(exist_ok=True)
                dest = pd / f"part_{k:03d}__{Path(names[k]).stem}{'.mp4' if ext == '.m4v' else ext}"
                tmp = dest.with_name(dest.name + ".tmp")                      # a half-received file is never mistaken for a whole one
                if not self._receive(tmp, n):
                    tmp.unlink(missing_ok=True)
                    return self._err(400, "upload interrupted - try that file again")
                tmp.replace(dest)
                have = len([p for p in pd.glob("part_*") if p.suffix != ".tmp"])
                if have >= len(names):
                    store.update(job["id"], status="draft", stage="Uploaded - set what you want, then press Clip it")
                    if projects:
                        projects.uploaded(job["id"])                          # starts it by itself if you asked for that
                else:
                    store.update(job["id"], stage=f"Uploading {have} of {len(names)}")
                return self._json(self._public(store.get(job["id"])), 201)
            opts_raw = json.loads((q.get("options") or ["{}"])[0])
            job = store.create({"filename": name}, clean_options(opts_raw, default_options(engine.ccfg)))
            dest = store.dir(job["id"]) / f"source{'.mp4' if ext == '.m4v' else ext}"
            store.update(job["id"], status="uploading", stage="Uploading", title=Path(name).stem)
            if not self._receive(dest, n):
                store.update(job["id"], status="failed", stage="Upload interrupted", error="upload interrupted")
                return self._err(400, "upload interrupted")
            store.update(job["id"], status="draft", stage="Uploaded - set what you want, then press Clip it")
            return self._json(self._public(store.get(job["id"])), 201)

        def _ai_post(self, parts: list[str]):
            b = self._body() if int(self.headers.get("Content-Length") or 0) else {}
            if parts == ["chatgpt-plan", "connect"]:
                if chatgpt_plan.status().get("status") == "connecting":
                    return self._err(409, "ChatGPT sign-in is already in progress")

                def connect():
                    try:
                        chatgpt_plan.connect()
                    except ChatGPTPlanError as e:
                        log.warning(f"ChatGPT plan sign-in failed: {e}")
                    except Exception:  # noqa: BLE001
                        log.exception("ChatGPT plan sign-in failed")
                threading.Thread(target=connect, daemon=True, name="clipfarm-chatgpt-signin").start()
                return self._json({"started": True}, 202)
            if parts == ["chatgpt-plan", "disconnect"]:
                try:
                    chatgpt_plan.disconnect()
                except ChatGPTPlanError as e:
                    return self._err(400, str(e))
                return self._json(chatgpt_plan.status())
            if parts == ["pick-folder"]:                         # a real macOS "choose a folder" window on the computer running the app
                if not (runner and runner.enabled()):
                    return self._err(403, "working on a folder of this computer is switched off here")
                import subprocess
                prompt = re.sub(r'["\\]', "", str(b.get("prompt") or "Choose the folder with your footage"))[:80]
                start = str(b.get("start") or "").replace('"', "")
                where = f' default location (POSIX file "{start}")' if start and Path(start).expanduser().is_dir() else ""
                try:
                    r = subprocess.run(["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}"{where})'],
                                       capture_output=True, text=True, timeout=600)
                except (subprocess.SubprocessError, OSError) as e:
                    return self._err(500, f"couldn't open the folder picker: {e}")
                return self._json({"path": r.stdout.strip().rstrip("/") if r.returncode == 0 else ""})
            if parts == ["projects"] and b.get("local_path"):
                if not (runner and runner.enabled()):
                    return self._err(403, "working on a folder of this computer is switched off here (clipper.ai_videos.local_projects)")
                proj = projects.create_local(str(b["local_path"]), str(b.get("title") or ""), str(b.get("notes") or ""), b,
                                             default_options(engine.ccfg), extra_audio=b.get("extra_audio"), out_dir=b.get("out_dir"))
                return self._json(projects.view(proj), 201)
            if parts == ["projects"]:
                proj = projects.create(str(b.get("title") or ""), str(b.get("notes") or ""), list(b.get("names") or []), b,
                                       default_options(engine.ccfg), auto_make=b.get("auto_make") or None)
                return self._json(projects.view(proj), 201)
            if parts == ["rules"]:
                if "text" in b:
                    projects.rules_path().write_text(str(b["text"]).strip()[:20000] + "\n")
                return self._json({"rules": projects.rules()})
            if len(parts) == 3 and parts[0] == "projects":
                pid, what = parts[1], parts[2]
                proj = projects.get(pid)
                if what == "resume":
                    return self._json(projects.resume(pid))
                if what == "notes":
                    return self._json(projects.view(projects.set_notes(pid, str(b.get("notes") or ""), b.get("title"))))
                if what == "make" and proj.get("mode") == "local":
                    projects.prepare_local(pid, b)
                    runner.start(pid, "make", body=b)
                    return self._json(projects.view(projects.get(pid)), 202)
                if what == "rerender" and proj.get("mode") == "local":       # the saved edit, rendered again: nothing is re-planned
                    if runner.running == pid:
                        return self._err(409, "it's being made right now")
                    runner.start(pid, "rerender")
                    return self._json(projects.view(projects.get(pid)), 202)
                if what == "approve-plan" and proj.get("mode") == "local":
                    if runner.running == pid:
                        return self._err(409, "it's being made right now")
                    if store.get(proj["job"])["status"] != "approval":
                        return self._err(409, "there is no script map waiting for approval")
                    runner.start(pid, "approve")
                    return self._json({"queued": True}, 202)
                if what == "plan-edit" and proj.get("mode") == "local":          # one click on the review screen: nothing renders
                    if runner.running == pid:
                        return self._err(409, "it's being made right now")
                    try:
                        runner.review_edit(pid, {k: b.get(k) for k in ("op", "first", "last", "sentence", "edge", "step") if k in b})
                    except ValueError as e:
                        return self._err(409, str(e))
                    return self._json(projects.view(projects.get(pid)))
                if what == "montage-edit" and proj.get("mode") == "local":      # one click on a montage's plan card: nothing renders
                    if runner.running == pid:
                        return self._err(409, "it's being made right now")
                    if b.get("op") == "normal":                                  # "Not what I meant": make it as an ordinary edit instead
                        job = store.get(proj["job"])
                        store.update(proj["job"], options={**job["options"], "montage": "off"})
                        runner.start(pid, "make")
                        return self._json({"queued": True}, 202)
                    try:
                        return self._json({"card": runner.montage_edit(pid, {k: b.get(k) for k in ("op", "id", "before", "value", "minutes") if k in b})})
                    except (ValueError, TypeError) as e:
                        return self._err(409, str(e))
                if what == "plan-find" and proj.get("mode") == "local":
                    query = str(b.get("query") or "").strip()[:300]
                    if not query:
                        return self._err(400, "say what you are looking for")
                    try:
                        return self._json({"candidates": runner.review_find(pid, int(b.get("step") or 0), query)})
                    except ValueError as e:
                        return self._err(409, str(e))
                if what == "revise-plan" and proj.get("mode") == "local":
                    text = str(b.get("instruction") or "").strip()[:1500]
                    if not text:
                        return self._err(400, "say what should use a different line")
                    if store.get(proj["job"])["status"] != "approval":
                        return self._err(409, "there is no script map waiting for revision")
                    runner.start(pid, "revise_plan", text=text)
                    return self._json({"queued": True}, 202)
                if what == "make":
                    return self._json(projects.view(projects.make(pid, b)), 202)
                if what == "revise" and proj.get("mode") == "local":
                    text = str(b.get("instruction") or "").strip()[:1500]
                    if not text and not b.get("revert"):
                        return self._err(400, "type what you want changed")
                    if store.get(proj["job"])["status"] != "done":
                        return self._err(409, "the video isn't finished yet")
                    runner.start(pid, "revise", text=text, revert=bool(b.get("revert")))
                    return self._json({"queued": True}, 202)
                if what == "revise":
                    text = str(b.get("instruction") or "").strip()[:800]
                    if not text:
                        return self._err(400, "type what you want changed")
                    job = store.get(proj["job"])
                    clip = (job.get("clips") or [None])[0]
                    if job["status"] != "done" or not clip:
                        return self._err(409, "the video isn't finished yet")
                    store.set_clip(job["id"], {**clip, "restyling": True, "edit_note": None})

                    def run():
                        try:
                            engine.edit_whole_video(job["id"], text, bool(b.get("revert")))
                            projects.finished(pid)
                        except Exception as e:  # noqa: BLE001
                            log.error(traceback.format_exc())
                            cur = store.get(job["id"])["clips"][0]
                            store.set_clip(job["id"], {**cur, "restyling": False, "edit_note": {
                                "instruction": text, "result": f"That didn't work: {str(e)[:200]}", "ok": False}})
                    with worker.lock:
                        worker.side_jobs.append(run)
                    worker.wake.set()
                    return self._json({"queued": True}, 202)
                if what == "feedback":
                    added = projects.learn(str(b.get("feedback") or ""), proj["title"])
                    return self._json({"added": added, "rules": projects.rules()})
                if what == "archive":
                    projects.update(pid, versions=[{**v, "archived": False} for v in proj["versions"]]) if b.get("again") else None
                    worker._last_ai_try = 0.0
                    return self._json({"result": projects.archive(pid)})
                if what == "cancel":
                    if proj.get("mode") == "local" and runner is not None and hasattr(runner, "hard_stop"):
                        return self._json({"ok": True, **runner.hard_stop(pid)})     # (asks, then kills what is stuck, then gives up on it)
                    store.update(proj["job"], cancel=True)
                    return self._json({"ok": True})
                if what in ("reveal", "open-fcp") and proj.get("mode") == "local":   # this Mac's Finder / Final Cut (the app's buttons)
                    import subprocess
                    home = Path(proj.get("home") or proj["folder"])
                    if what == "reveal":                                      # the finished video selected in Finder, else the folder
                        vids = sorted((home / "Output").glob("*.mp4"), key=lambda x: x.stat().st_mtime) if (home / "Output").is_dir() else []
                        if not home.is_dir():
                            return self._err(404, "that folder isn't there any more")
                        subprocess.run(["open", "-R", str(vids[-1])] if vids else ["open", str(home)], capture_output=True, timeout=30)
                        return self._json({"ok": True})
                    xmls = sorted(home.glob("*.fcpxml"), key=lambda x: x.stat().st_mtime)
                    if not xmls:
                        return self._err(404, "no Final Cut project yet")
                    subprocess.run(["open", str(xmls[-1])], capture_output=True, timeout=30)
                    return self._json({"ok": True, "path": str(xmls[-1])})
            return self._err(404, "not found")

        def _edit(self, job_id: str, index: int, b: dict):
            """Tell the AI what to change in a finished clip (or revert to the first cut)."""
            job = store.get(job_id)
            clip = next((c for c in job["clips"] if c["index"] == index), None)
            if clip is None:
                return self._err(404, "no such clip")
            revert = bool(b.get("revert"))
            text = str(b.get("instruction") or "").strip()[:800]
            if not revert and not text:
                return self._err(400, "type what you want changed")
            store.set_clip(job_id, {**clip, "restyling": True, "edit_note": None})

            def run():
                try:
                    if clip.get("polish"):
                        engine.edit_whole_video(job_id, text, revert)
                    else:
                        engine.edit_clip(job_id, index, text, revert)
                except Exception as e:  # noqa: BLE001
                    log.error(traceback.format_exc())
                    cur = next(c for c in store.get(job_id)["clips"] if c["index"] == index)
                    store.set_clip(job_id, {**cur, "restyling": False,
                                            "edit_note": {"instruction": text, "result": f"That didn't work: {str(e)[:200]}", "ok": False}})
            with worker.lock:
                worker.side_jobs.append(run)
            worker.wake.set()
            return self._json({"queued": True}, 202)

        def _restyle(self, job_id: str, index: int, b: dict):
            job = store.get(job_id)
            clip = next((c for c in job["clips"] if c["index"] == index), None)
            if clip is None:
                return self._err(404, "no such clip")
            layout = b.get("layout") if b.get("layout") in LAYOUTS else clip["layout"]
            style = b.get("caption_style") if b.get("caption_style") in CAPTION_STYLES else clip["caption_style"]
            pace = b.get("pace") if b.get("pace") in PACES else clip["pace"]
            hook = bool(b.get("hook_title", clip.get("hook_title", True)))
            title = b.get("title") if isinstance(b.get("title"), str) else None
            estyle = b.get("edit_style") if b.get("edit_style") in ("standard", "auto", "youtube", "business") else None
            music = None
            if "music" in b:
                mv = b.get("music")
                music = "" if mv in (False, None, "", "off") else ("auto" if mv is True else str(mv)[:20])
            store.set_clip(job_id, {**clip, "restyling": True})

            def run():
                try:
                    engine.restyle(job_id, index, layout, style, pace, hook, title, music, edit_style=estyle)
                except Exception as e:  # noqa: BLE001
                    store.set_clip(job_id, {**clip, "restyling": False, "restyle_error": str(e)[:300]})
            with worker.lock:
                worker.side_jobs.append(run)
            worker.wake.set()
            return self._json({"queued": True}, 202)

    return Handler


def serve(cfg, host: str = "127.0.0.1", port: int = 8799) -> None:
    ccfg = cfg.get("clipper", {}) or {}
    store = Store(cfg.root / ccfg.get("data_dir", "data/clipper"))
    engine = Engine(cfg, store)
    if ccfg.get("app") == "studio":                     # the app: short clips ask the AI the person chose (their plan, or this Mac)
        from .localai import LocalAI
        from .localproj import LocalEngine
        engine = LocalEngine(cfg, store, LocalAI(cfg))
    feed = Feed(cfg, store)
    projects = aivideo.Projects(cfg, store, engine)
    from .localproj import LocalRunner
    runner = LocalRunner(cfg, store, projects)
    worker = Worker(engine, store, feed, projects)
    worker.start()
    if ccfg.get("app") == "studio":                     # the app: a long video that was being made when it quit carries on
        runner.keep_resuming()
    cat = runner.library()
    if cat is not None:                                  # the footage catalog: ready to search, and catalogues by itself at night
        cat.busy = lambda: bool(runner.running or worker.busy_with)
        cat.looping = True                               # (this loop ticks it; a catalog made later by the app is ticked by studio_web)

        def catalog_loop():
            try:
                cat.embedder()                           # load the picture model now, so the first search is quick
            except Exception:  # noqa: BLE001 - search says why when it is used
                log.error(traceback.format_exc())
            while True:
                try:
                    cat.tick(busy=cat.busy())
                except Exception:  # noqa: BLE001
                    log.error(traceback.format_exc())
                time.sleep(300)
        threading.Thread(target=catalog_loop, name="catalog-loop", daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, store, worker, feed,
                                                           float(ccfg.get("max_upload_mb", 8000)), projects, runner))
    httpd.daemon_threads = True
    shown = "localhost" if host in ("127.0.0.1", "localhost") else host
    print(f"Clipper running at http://{shown}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def run_once(cfg, url: str | None, file: str | list[str] | None, options: dict) -> dict:
    """Headless: process one video synchronously (CLI)."""
    ccfg = cfg.get("clipper", {}) or {}
    store = Store(cfg.root / ccfg.get("data_dir", "data/clipper"))
    engine = Engine(cfg, store)
    files = [file] if isinstance(file, str) else list(file or [])
    src: dict = {}
    if url:
        src["url"] = url
    elif len(files) == 1:
        src["filename"] = Path(files[0]).name
    else:
        src["files"] = [Path(f).name for f in files]
    job = store.create(src, clean_options(options, default_options(ccfg)))
    if len(files) == 1:
        shutil.copy2(files[0], store.dir(job["id"]) / ("source" + (Path(files[0]).suffix.lower() or ".mp4")))
        store.update(job["id"], title=Path(files[0]).stem)
    elif files:                                           # several files -> one joined timeline
        pd = store.dir(job["id"]) / "parts"
        pd.mkdir()
        for k, f in enumerate(sorted(files)):
            shutil.copy2(f, pd / f"part_{k:03d}__{Path(f).name}")
        store.update(job["id"], title=f"{len(files)} videos: {Path(sorted(files)[0]).stem} ...")
    t0 = time.monotonic()
    job = engine.process(job["id"])
    job["wall_s"] = round(time.monotonic() - t0, 1)
    job["dir"] = str(store.dir(job["id"]))
    return job
