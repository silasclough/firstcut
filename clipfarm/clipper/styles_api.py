"""The app's API for editing-style presets (Settings › How it edits › "Learn from a link"). Kept out of studio_web.py: that file calls `get` and `post` here.

    GET  /api/app/styles/presets                         every preset (what the page shows), and the fallback address setting
    GET  /api/app/styles/learn/<job>                     how a learning job is doing: {state: running|done|error, text, preset?, error?}
    POST /api/app/styles/learn {url | file, preset_id?, name?}     learn from a link (or a dropped file) into a new preset, or add it to preset_id
    POST /api/app/styles/presets {name?, points?, rules?}          a new empty preset
    POST /api/app/styles/presets/<id>/rename {name} | delete | points {points} | reset-points | rule {text} | remove-rule {index}
    POST /api/app/styles/presets/<id>/apply {mode: short|long}     fill that mode's points box and settings with the preset ("Use this")

Learning is one background thread per job (polled); at most two run at once. Nothing is applied to how videos are edited until `apply`.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path

from ..core.log import get_logger
from . import studio as SD
from . import stylefetch, stylelearn, stylepresets
from . import styles_profile as SPF

log = get_logger("clipper.styles_api")
MAX_RUNNING = 2
KEEP_JOBS = 20


def presets(api) -> stylepresets.Presets:
    p = getattr(api, "_presets", None)
    if p is None:
        p = api._presets = stylepresets.Presets(api.root)
    return p


def _jobs(api) -> dict:
    j = getattr(api, "_style_jobs", None)
    if j is None:
        j = api._style_jobs = {"lock": threading.Lock(), "by_id": {}}
    return j


def summary(api) -> dict:
    """What the page needs in its main state: the presets, any learning job still running."""
    jobs = _jobs(api)
    with jobs["lock"]:
        running = [{"id": k, "text": v["text"], "preset_id": v.get("preset_id")} for k, v in jobs["by_id"].items() if v["state"] == "running"]
    return {"presets": presets(api).summary(), "running": running, "fetch_url": ((api.settings.get().get("styles") or {}).get("fetch_url") or "")}


def _set(api, job_id: str, **kw) -> None:
    jobs = _jobs(api)
    with jobs["lock"]:
        if job_id in jobs["by_id"]:
            jobs["by_id"][job_id].update(kw)


def start_learning(api, source: str, preset_id: str | None = None, name: str = "", ai=None) -> dict:
    """Start a learning job (a thread). -> {"job": id}; ValueError for a plain problem with the request."""
    source = str(source or "").strip()
    kind = stylefetch.classify(source)
    if kind == "other":
        raise ValueError(stylefetch.NEED_VIDEO)
    if kind == "file":
        p = Path(source[7:] if source.startswith("file://") else source).expanduser()
        if not p.is_absolute() or not p.is_file() or p.suffix.lower() not in stylefetch.VIDEO_EXT:
            raise ValueError("that file is not a video I can open")
        source = str(p)
    if preset_id:
        presets(api).get(preset_id)                                           # KeyError -> "no such style"
    jobs = _jobs(api)
    with jobs["lock"]:
        if sum(1 for v in jobs["by_id"].values() if v["state"] == "running") >= MAX_RUNNING:
            raise ValueError("two styles are already being learned: wait for one to finish")
        job_id = "sl" + uuid.uuid4().hex[:8]
        jobs["by_id"][job_id] = {"state": "running", "text": "Starting", "started": time.time(), "preset_id": preset_id, "error": "", "preset": None}
        for old in sorted(jobs["by_id"], key=lambda k: jobs["by_id"][k]["started"])[:-KEEP_JOBS]:
            jobs["by_id"].pop(old, None)

    def work() -> None:
        try:
            engine_ai = ai
            if engine_ai is None:
                try:
                    from .localai import LocalAI
                    engine_ai = LocalAI(api.cfg)
                except Exception as e:  # noqa: BLE001 - the points are then written from the numbers
                    log.info(f"no AI for the style points: {e}")
            endpoint = (api.settings.get().get("styles") or {}).get("fetch_url") or ""
            frames_dir = presets(api).dir / "_incoming" / job_id                    # (its own folder: another job running at the same time keeps its frames)
            res = stylelearn.learn(source, ai=engine_ai, endpoint=endpoint, progress=lambda m: _set(api, job_id, text=m), frames_dir=frames_dir)
            _set(api, job_id, text="Writing the style")
            title = res["sources"][0].get("title") or "Learned style"
            view = presets(api).add_learned(preset_id, name or title, res["sources"], res["frames"], engine_ai)
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
            try:
                (presets(api).dir / "_incoming").rmdir()                         # (only goes if no other job still has frames there)
            except OSError:
                pass
            _set(api, job_id, state="done", text="Done", preset=view)
        except stylefetch.FetchError as e:
            _set(api, job_id, state="error", text="", error=str(e))
        except KeyError:
            _set(api, job_id, state="error", text="", error="that style no longer exists")
        except Exception as e:  # noqa: BLE001
            log.warning(f"learning a style from {source[:80]} failed: {e}")
            _set(api, job_id, state="error", text="", error="Something went wrong while learning from that video. Try another one.")
    threading.Thread(target=work, daemon=True, name="style-learn").start()
    return {"job": job_id}


def job_view(api, job_id: str) -> dict | None:
    jobs = _jobs(api)
    with jobs["lock"]:
        j = jobs["by_id"].get(job_id)
        return None if j is None else {"id": job_id, "state": j["state"], "text": j["text"], "error": j["error"], "preset": j["preset"],
                                       "seconds": round(time.time() - j["started"])}


def apply_preset(api, pid: str, mode: str) -> dict:
    """"Use this": the preset's rules and lines become the points of `mode` (short|long), and its settings the settings those points state. The person can
    still edit them in the box. Hard rules of the app are not touched: a style only sets editing numbers and soft wishes."""
    if mode not in SD.KINDS:
        raise ValueError("choose short clips or long videos")
    pr = presets(api)
    p = pr.get(pid)
    lines = pr.lines_for(p)
    if not lines:
        raise ValueError("that style has no lines yet: learn from a video or write some first")
    settings = pr.settings_for(p)
    api.styles.replace(mode, lines)
    api.settings.update({"style_overrides": {mode: {k: settings.get(k) for k in SPF.SETTING_KEYS}}})
    if mode == "long":
        api.sync_long_style(settings)
    else:
        api.sync_points_style(mode, lines, settings)
    return {"ok": True, "mode": mode, "points": lines, "settings": settings, "styles": {k: api.style_view(k) for k in SD.KINDS}}


def get(api, rest: list[str]) -> tuple[int, dict] | None:
    """GET routes under styles/ ('presets' and 'learn'); None for any other path."""
    if rest == ["presets"]:
        return 200, summary(api)
    if len(rest) == 2 and rest[0] == "learn" and re.fullmatch(r"sl[0-9a-f]{8}", rest[1]):
        v = job_view(api, rest[1])
        return (200, v) if v else (404, {"error": "no such job"})
    return None


def post(api, rest: list[str], b: dict) -> tuple[int, dict] | None:
    """POST routes under styles/ ('presets' and 'learn'); None for any other path. A plain problem is a 400 with the words for the person."""
    pr = presets(api)
    try:
        if rest == ["learn"]:
            src = b.get("file") or b.get("url") or ""
            if not str(src).strip():
                raise ValueError("paste a link or drop a video first")
            return 200, start_learning(api, str(src), str(b.get("preset_id") or "") or None, str(b.get("name") or ""))
        if rest == ["presets"]:
            return 200, {"preset": pr.create(str(b.get("name") or ""), b.get("points"), b.get("rules")), **summary(api)}
        if len(rest) == 3 and rest[0] == "presets":
            pid, what = rest[1], rest[2]
            if what == "rename":
                pr.rename(pid, str(b.get("name") or ""))
            elif what == "delete":
                pr.delete(pid)
            elif what == "points":
                pr.set_points(pid, b.get("points"))
            elif what == "reset-points":
                pr.reset_points(pid)
            elif what == "rule":
                pr.add_rule(pid, str(b.get("text") or ""))
            elif what == "remove-rule":
                pr.remove_rule(pid, int(b.get("index", -1)))
            elif what == "apply":
                return 200, {**apply_preset(api, pid, str(b.get("mode") or "")), **summary(api)}
            else:
                return None
            return 200, summary(api)
    except KeyError as e:
        return 404, {"error": str(e.args[0]) if e.args else "no such style"}
    except (ValueError, TypeError) as e:
        return 400, {"error": str(e) or "that did not work"}
    return None
