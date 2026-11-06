"""Public FirstCut build: the posting relay (moving liked clips to a posting server, its review queue) is not part of the app.
The server's code still names these; in the app they are never reached (its background loop is off), and they answer "nothing"."""

from __future__ import annotations


def ingest(ctx) -> list[str]:
    return []


def apply_verdicts(ctx) -> int:
    return 0


def auto_approve(ctx) -> list[str]:
    return []


def clip_states(ctx, job: dict, info=None) -> dict:
    """Posting state per clip index: in the app every clip is simply as it was reviewed."""
    return {c["index"]: {"state": (c.get("review") or {}).get("state", ""), "why": (c.get("review") or {}).get("why")}
            for c in job.get("clips", []) if "index" in c}


def _no(*_a, **_k):
    raise RuntimeError("posting isn't part of FirstCut")


approve_clip = reject_clip = discard_clip = queue_own = _no
