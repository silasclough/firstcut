"""Not part of FirstCut: the server's review feed for clips. Every call answers "nothing"; `local` is False, so none of the server's
background work that depends on it ever runs."""

from __future__ import annotations


class Feed:
    local = False

    def __init__(self, cfg=None, store=None, *a, **k):
        self.cfg, self.store, self._prev_failed = cfg, store, set()

    def _key(self, jid, idx):
        return f"{jid}:{idx}"

    def items(self, *a, **k) -> list:
        return []

    def stats(self, *a, **k) -> dict:
        return {}

    def queue_info(self, *a, **k) -> dict:
        return {}

    def html(self, *a, **k) -> str:
        return ""

    def next_stale(self, *a, **k):
        return None

    def commentate_backlog_one(self, *a, **k) -> bool:
        return False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None
