"""Public FirstCut build: the server's older Long-form review tab (/longform) belongs to a channel pipeline that isn't part of the
app. These answer plainly when something asks."""

from __future__ import annotations


class LongformWeb:
    def __init__(self, cfg):
        self.cfg = cfg

    def payload(self) -> dict:
        return {"error": "not part of FirstCut", "ideas": [], "videos": []}

    def media(self, rel: str):
        return None

    def post(self, parts: list[str], b: dict) -> tuple[dict, int]:
        return {"error": "not part of FirstCut"}, 404
