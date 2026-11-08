"""Not part of FirstCut: the shared context of the channel's posting pipelines."""

from __future__ import annotations


class Context:
    def __init__(self, cfg=None, *a, **k):
        self.cfg = cfg
