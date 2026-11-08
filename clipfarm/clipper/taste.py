"""Not part of FirstCut: learning from clip ratings. It adds nothing to a clip's score."""

from __future__ import annotations


class Taste:
    def __init__(self, *a, **k):
        pass

    def moment_bonus(self, *a, **k) -> float:
        return 0.0
