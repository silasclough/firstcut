"""Shared runtime context passed to pipelines, publishers and CLI commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .core.config import Config, load_config
from .core.dedupe import Deduper
from .core.llm import LLMRouter
from .core.log import setup_logging
from .core.queue import Job, JobQueue
from .core.state import JsonState


@dataclass
class Context:
    cfg: Config
    dry_run: bool = False
    queue: JobQueue = field(init=False)
    dedupe: Deduper = field(init=False)
    llm: LLMRouter = field(init=False)
    usage: JsonState = field(init=False)

    def __post_init__(self) -> None:
        if not self.dry_run:
            self.cfg.ensure_dirs()
        self.queue = JobQueue(self.cfg.path("queue"), dry_run=self.dry_run)
        self.dedupe = Deduper(self.cfg.path("dedupe_db"), dry_run=self.dry_run)
        self.llm = LLMRouter(self.cfg.get("llm", {}), dry_run=self.dry_run)
        self.should_stop = lambda: False      # the night runner swaps in its deadline/signal/power check
        self.deadline = None                  # ...and its stop time, so long stages don't start what won't finish
        self.usage = JsonState(self.cfg.path("state") / "asset_usage.json", dry_run=self.dry_run)

    def work_dir(self, job: Job) -> Path:
        d = self.cfg.path("work") / job.id
        if not self.dry_run:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, p: Path | str) -> str:
        """Store paths relative to the project root so the project folder can move machines."""
        p = Path(p)
        try:
            return str(p.resolve().relative_to(self.cfg.root.resolve()))
        except ValueError:
            return str(p)

    def abs(self, p: Path | str) -> Path:
        return self.cfg.resolve(p)


def make_context(dry_run: bool = False, verbose: bool = False, quiet: bool = False,
                 config_path: str | None = None) -> Context:
    cfg = load_config(config_path)
    setup_logging(None if dry_run else cfg.path("logs"), verbose=verbose, quiet=quiet)
    return Context(cfg, dry_run=dry_run)
