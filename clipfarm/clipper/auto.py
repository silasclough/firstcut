"""Not part of FirstCut: the nightly clipper of licensed sources. These answer "nothing"."""

from __future__ import annotations


def _target(cfg):
    return None


class Pace:
    def __init__(self, cfg=None):
        pass

    def ok(self):
        return False, "not part of FirstCut"


def run_auto(*a, **k) -> list:
    return []
