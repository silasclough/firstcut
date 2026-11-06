"""On-disk job queue with an explicit state machine.

Each job is one JSON file in data/queue/<id>.json, written atomically (tmp + rename +
fsync), so a crash can never leave a half-written job. Stages are idempotent and keyed
off the job's state, so after a crash the runner just picks every non-terminal job up
from the state it was last saved in.

States (the six you asked for, plus `approved` and `rejected` so review is explicit):

    idea -> scripted -> rendered -> review -> approved -> scheduled -> posted
                                  \\-> approved (auto_approve)
    review -> rejected
    any non-terminal -> failed ;  failed -> (retry) last good state
    scheduled -> approved   (unschedule / missed manual slot goes back to the pool)
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator

from .log import get_logger
from .state import atomic_write_json, read_json

log = get_logger("queue")


class State(str, Enum):
    IDEA = "idea"
    SCRIPTED = "scripted"
    RENDERED = "rendered"
    REVIEW = "review"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    POSTED = "posted"
    FAILED = "failed"
    REJECTED = "rejected"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


TRANSITIONS: dict[State, set[State]] = {
    State.IDEA: {State.SCRIPTED, State.FAILED},
    State.SCRIPTED: {State.RENDERED, State.FAILED, State.IDEA},
    State.RENDERED: {State.REVIEW, State.APPROVED, State.FAILED},
    State.REVIEW: {State.APPROVED, State.REJECTED, State.FAILED, State.SCRIPTED},   # -> scripted = re-render
    State.APPROVED: {State.SCHEDULED, State.REVIEW, State.REJECTED, State.FAILED},
    State.SCHEDULED: {State.POSTED, State.APPROVED, State.FAILED},
    State.POSTED: set(),
    State.REJECTED: set(),
    State.FAILED: {State.IDEA, State.SCRIPTED, State.RENDERED, State.REVIEW, State.APPROVED,
                   State.SCHEDULED, State.REJECTED},
}
TERMINAL = {State.POSTED, State.REJECTED}
# States the generation runner advances on its own.
GENERATION_STATES = (State.IDEA, State.SCRIPTED, State.RENDERED)


class InvalidTransition(Exception):
    pass


class AlreadyRunning(Exception):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    format: str
    brand: str
    state: State = State.IDEA
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    data: dict = field(default_factory=dict)        # format-specific payload (idea, story, clip...)
    artifacts: dict = field(default_factory=dict)   # file paths produced by stages
    publish: dict = field(default_factory=dict)     # slot, publisher ids, status
    attempts: dict = field(default_factory=dict)    # stage -> attempt count
    history: list = field(default_factory=list)     # transitions
    error: str | None = None
    last_good_state: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        d = dict(d)
        d["state"] = State(d.get("state", "idea"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def title(self) -> str:
        return (self.data.get("story", {}) or {}).get("title") or self.data.get("title") or self.id


def can_transition(src: State, dst: State) -> bool:
    return src == dst or dst in TRANSITIONS[src]


class JobQueue:
    PREFIX = {"minecraft": "mc", "streamer": "st", "aigen": "ai", "clipper": "cl"}

    def __init__(self, directory: Path, dry_run: bool = False):
        self.dir = Path(directory)
        self.dry_run = dry_run
        self._mem: dict[str, Job] = {}      # dry-run jobs live only in memory
        if not dry_run:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ storage
    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def new_id(self, fmt: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.PREFIX.get(fmt, fmt[:2])}-{stamp}-{secrets.token_hex(2)}"

    def save(self, job: Job) -> None:
        job.updated_at = utcnow()
        if self.dry_run:
            self._mem[job.id] = job
            return
        atomic_write_json(self._path(job.id), job.to_dict())

    def get(self, job_id: str) -> Job:
        if job_id in self._mem:
            return self._mem[job_id]
        d = read_json(self._path(job_id))
        if d is None:
            matches = [p for p in self.dir.glob(f"{job_id}*.json")] if self.dir.exists() else []
            if len(matches) == 1:
                d = read_json(matches[0])
            else:
                raise KeyError(f"job {job_id} not found" + (" (ambiguous prefix)" if matches else ""))
        return Job.from_dict(d)

    def all(self) -> list[Job]:
        jobs: dict[str, Job] = {}
        if self.dir.exists():
            for p in self.dir.glob("*.json"):
                if p.name.startswith("."):
                    continue
                try:
                    jobs[p.stem] = Job.from_dict(read_json(p))
                except Exception as e:  # noqa: BLE001 - one corrupt file must not stop the queue
                    log.error(f"unreadable job file {p.name}: {e}")
        jobs.update(self._mem)
        return sorted(jobs.values(), key=lambda j: (j.created_at, j.id))

    def list(self, state: State | str | None = None, fmt: str | None = None,
             brand: str | None = None) -> list[Job]:
        st = State(state) if state else None
        return [j for j in self.all()
                if (st is None or j.state == st) and (fmt is None or j.format == fmt)
                and (brand is None or j.brand == brand)]

    def create(self, fmt: str, brand: str, data: dict | None = None) -> Job:
        job = Job(id=self.new_id(fmt), format=fmt, brand=brand, data=data or {})
        job.history.append({"at": utcnow(), "to": State.IDEA.value, "note": "created"})
        self.save(job)
        log.info(f"created job {job.id}", extra={"job": job.id, "fmt": fmt})
        return job

    # ------------------------------------------------------------ state machine
    def transition(self, job: Job, to: State | str, note: str = "") -> Job:
        to = State(to)
        if job.state == to:
            return job  # idempotent
        if not can_transition(job.state, to):
            raise InvalidTransition(f"{job.id}: {job.state.value} -> {to.value} not allowed")
        job.history.append({"at": utcnow(), "from": job.state.value, "to": to.value, "note": note})
        if to != State.FAILED:
            job.error = None
            job.last_good_state = to.value
        job.state = to
        self.save(job)
        log.info(f"{job.id}: -> {to.value}" + (f" ({note})" if note else ""),
                 extra={"job": job.id, "fmt": job.format, "event": "transition", "to": to.value})
        return job

    def record_attempt(self, job: Job, stage: str) -> int:
        job.attempts[stage] = int(job.attempts.get(stage, 0)) + 1
        self.save(job)
        return job.attempts[stage]

    def fail(self, job: Job, stage: str, error: str) -> Job:
        if job.state != State.FAILED:
            job.last_good_state = job.state.value
        job.error = f"[{stage}] {error}"[:2000]
        job.history.append({"at": utcnow(), "from": job.state.value, "to": State.FAILED.value,
                            "note": f"{stage}: {error}"[:500]})
        job.state = State.FAILED
        self.save(job)
        log.error(f"{job.id} failed at {stage}: {error}",
                  extra={"job": job.id, "fmt": job.format, "stage": stage, "event": "failed"})
        return job

    def retry(self, job: Job, reset_attempts: bool = True) -> Job:
        if job.state != State.FAILED:
            raise InvalidTransition(f"{job.id} is {job.state.value}, not failed")
        target = State(job.last_good_state or State.IDEA.value)
        if reset_attempts:
            job.attempts = {}
        return self.transition(job, target, note="retry")

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for j in self.all():
            out.setdefault(j.format, {}).setdefault(j.state.value, 0)
            out[j.format][j.state.value] += 1
        return out

    # ------------------------------------------------------------------- locking
    @contextlib.contextmanager
    def lock(self, name: str = "runner") -> Iterator[None]:
        """Process-exclusive lock so two night runs (or a run + a post) never overlap."""
        if self.dry_run:
            yield
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f".{name}.lock"
        fh = open(path, "a+")
        try:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise AlreadyRunning(f"another clipfarm '{name}' process holds {path}") from e
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
