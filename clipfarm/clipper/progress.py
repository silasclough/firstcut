"""How far along a video or a batch of clips is, in time, not in a fixed percentage.

The old bar jumped between hard-coded fractions ("Saving into the folder" = 97%), so one slow step (an AI call, a big copy) left it
sitting still for minutes. Here every run's stages are timed and remembered on this Mac (a small JSON file, the last ~50 runs); the bar
is "time spent against the time this Mac usually needs", so it moves while a step works, eases off (never stops, never reaches 100%)
when a step takes longer than usual, and says how long is left only when there is history to say it from.

The funnel is Store.update: every job (long form and short clips) already goes through it with status / stage changes, so one listener
sees both without the engines knowing about it.

    WATCH.on_update(job_id, job, fields)      (registered on the Store by the studio)
    WATCH.view(job)                           -> the extra fields the page shows: progress, eta_s, stage_elapsed_s, stage_expected_s,
                                                 long, stage_key, stages_done
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.progress")

ACTIVE = ("downloading", "transcribing", "finding", "rendering", "joining", "running")
WORDS = {"downloading": "Downloading", "transcribing": "Transcribing", "finding": "Finding footage and planning", "rendering": "Rendering",
         "joining": "Joining the clips", "running": "Working"}
KEEP_RUNS = 50
MIN_RUNS_FOR_ETA = 2          # "about 6 min left" needs at least this many finished runs of the same kind
LONG_FACTOR = 1.3             # a stage is "taking longer than usual" past this share of its usual time ...
LONG_MARGIN_S = 20.0          # ... and this many seconds over
CAP = 0.99                    # never "done" until it is


def kind_of(job: dict) -> str:
    """long:make / long:rerender / ... for a project's job; short for a batch of clips."""
    run = job.get("run") or {}
    if run.get("kind"):
        return "long:" + str(run["kind"])
    if (job.get("options") or {}).get("project"):
        return "long:make"
    return "short"


def _sane_run(r) -> bool:
    """A remembered run we can use: a dict whose stages map names to numbers (anything else, hand-edited or damaged, is ignored)."""
    if not isinstance(r, dict):
        return False
    u = r.get("units")
    if u is not None and (isinstance(u, bool) or not isinstance(u, (int, float))):
        return False
    st = r.get("stages")
    if st is None:
        return True
    return isinstance(st, dict) and all(isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in st.items())


class History:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.runs: list[dict] = []
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.runs = [r for r in (raw if isinstance(raw, list) else []) if _sane_run(r)][-KEEP_RUNS:]
            except (OSError, ValueError, TypeError):
                self.runs = []

    def record(self, run: dict) -> None:
        with self.lock:
            self.runs = (self.runs + [run])[-KEEP_RUNS:]
            if self.path:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.path.with_suffix(".part")
                    tmp.write_text(json.dumps(self.runs))
                    tmp.replace(self.path)
                except OSError as e:
                    log.warning(f"progress history not saved: {e}")

    def of_kind(self, kind: str) -> list[dict]:
        with self.lock:
            return [r for r in self.runs if r.get("kind") == kind and isinstance(r.get("stages"), dict) and r.get("stages")]

    def expected(self, kind: str, units: float | None) -> dict[str, float]:
        """{stage key: usual seconds} for this kind of run (median; scaled by size when both the past runs and this one know it)."""
        runs = self.of_kind(kind)
        out: dict[str, float] = {}
        keys = []
        for r in runs:
            for k in r["stages"]:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            secs = [(float(r["stages"][k]), float(r.get("units") or 0)) for r in runs if k in r["stages"]]
            rates = [s / u for s, u in secs if u > 0]
            if units and units > 0 and len(rates) >= 3:
                out[k] = max(1.0, statistics.median(rates) * units)
            else:
                out[k] = max(1.0, statistics.median(s for s, _ in secs))
        return out

    def order(self, kind: str) -> list[str]:
        runs = self.of_kind(kind)
        return list(runs[-1]["stages"]) if runs else []


class Run:
    def __init__(self, kind: str, units: float | None, now: float):
        self.kind, self.units, self.t0 = kind, units, now
        self.cur: dict | None = None                    # {key, label, start, f0, f}
        self.done: list[dict] = []                      # [{key, label, seconds, f0, f1}]
        self.bar = 0.0                                  # the highest value shown: the bar never goes back


class Watch:
    def __init__(self, history: History | None = None, clock=time.time):
        self.history = history or History(None)
        self.clock = clock
        self.runs: dict[str, Run] = {}
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- the Store.update listener
    def on_update(self, job_id: str, job: dict, fields: dict) -> None:
        if not ({"status", "stage", "progress"} & set(fields)):
            return
        try:
            self._on_update(job_id, job, fields)
        except Exception as e:  # noqa: BLE001 - a progress note never breaks a video
            log.warning(f"progress watch: {e}")

    def _on_update(self, job_id: str, job: dict, fields: dict) -> None:
        now = self.clock()
        status = str(job.get("status") or "")
        with self.lock:
            run = self.runs.get(job_id)
            if status in ACTIVE:
                if run is None:
                    run = self.runs[job_id] = Run(kind_of(job), self._units(job), now)
                f = float(job.get("progress") or 0.0)
                c = run.cur
                if c is None or c["key"] != status:
                    self._close(run, now, f)
                    run.cur = {"key": status, "label": job.get("stage") or WORDS.get(status, status), "start": now, "f0": f, "f": f}
                else:
                    c["label"] = job.get("stage") or c["label"]
                    c["f"] = max(c["f"], f)
            elif status == "done" and run is not None:
                self._close(run, now, float(job.get("progress") or 1.0))
                stages: dict[str, float] = {}
                for d in run.done:
                    stages[d["key"]] = stages.get(d["key"], 0.0) + d["seconds"]
                if stages and now - run.t0 >= 3:
                    self.history.record({"kind": run.kind, "at": int(now), "units": run.units, "total_s": round(now - run.t0, 1),
                                         "stages": {k: round(v, 1) for k, v in stages.items()}})
                self.runs.pop(job_id, None)
            elif status in ("failed", "cancelled", "draft", "queued") and run is not None:
                self.runs.pop(job_id, None)

    def _close(self, run: Run, now: float, f: float) -> None:
        c = run.cur
        if c is not None:
            run.done.append({"key": c["key"], "label": c["label"], "seconds": max(0.0, now - c["start"]), "f0": c["f0"], "f1": max(c["f"], f)})
            run.cur = None

    def _units(self, job: dict) -> float | None:
        try:
            u = float((job.get("run") or {}).get("units") or 0) or float((job.get("source") or {}).get("minutes") or 0)
            return u or None
        except (TypeError, ValueError):
            return None

    def set_units(self, job_id: str, units: float | None) -> None:
        with self.lock:
            r = self.runs.get(job_id)
            if r is not None and units and units > 0:
                r.units = float(units)

    # ---------------------------------------------------------------- what the page shows
    def view(self, job: dict) -> dict:
        """Extra fields for a job's state. Empty when nothing is running (the page then uses the stored progress as it is)."""
        status = str(job.get("status") or "")
        if status not in ACTIVE:
            return {}
        with self.lock:
            run = self.runs.get(job.get("id") or "")
            if run is None or run.cur is None:
                return {}
            return self._view(run, float(job.get("progress") or 0.0), self.clock())

    def _view(self, run: Run, static: float, now: float) -> dict:
        c = run.cur
        elapsed = max(0.0, now - c["start"])
        hist = self.history.of_kind(run.kind)
        exp = self.history.expected(run.kind, run.units) if hist else {}
        out = {"stage_key": c["key"], "stage_elapsed_s": round(elapsed), "stage_expected_s": None, "eta_s": None, "long": False,
               "stages_done": [{"key": d["key"], "label": WORDS.get(d["key"], d["key"]), "seconds": round(d["seconds"])} for d in run.done]}
        if not exp or c["key"] not in exp:
            # no history for this stage: the stored value (fixed fractions), never going back and never 100% before it is done
            run.bar = max(run.bar, min(CAP, static))
            out["progress"] = round(run.bar, 4)
            return out
        e_cur = exp[c["key"]]
        out["stage_expected_s"] = round(e_cur)
        order = self.history.order(run.kind)
        seen = {d["key"] for d in run.done} | {c["key"]}
        ahead = [k for k in order if k not in seen]
        total = sum(exp[k] for k in seen if k in exp) + sum(exp[k] for k in ahead)
        before = sum(exp[d["key"]] for d in run.done if d["key"] in exp)
        if elapsed <= e_cur:
            within = 0.95 * elapsed / e_cur
        else:                                                      # longer than usual: the bar eases towards the end of the stage, never sits
            within = 0.95 + 0.04 * (1 - math.exp(-(elapsed - e_cur) / e_cur))
        frac = (before + within * e_cur) / total if total > 0 else static
        frac = min(CAP, max(frac, 0.0))
        run.bar = max(run.bar, frac)
        out["progress"] = round(run.bar, 4)
        out["long"] = elapsed > e_cur * LONG_FACTOR and elapsed - e_cur > LONG_MARGIN_S
        if len(hist) >= MIN_RUNS_FOR_ETA and not out["long"]:
            left = max(e_cur - elapsed, 0.1 * e_cur) + sum(exp[k] for k in ahead)
            out["eta_s"] = int(round(left))
        return out


WATCH = Watch()


def configure(home: Path) -> Watch:
    """Called once by the studio: history lives in the app home."""
    WATCH.history = History(Path(home) / "progress_history.json")
    return WATCH


def clock_text(seconds) -> str:
    s = int(max(0, round(float(seconds or 0))))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
