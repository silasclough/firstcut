"""Licensed music library: audio files in assets/music/ with a sidecar JSON each.

Sidecar (same basename, .json), all fields required:
    {"title": "...", "artist": "Kevin MacLeod", "license": "CC BY 4.0",
     "credit": "\"Title\" Kevin MacLeod (incompetech.com) Licensed under Creative Commons: By Attribution 4.0 License http://creativecommons.org/licenses/by/4.0/",
     "source_url": "https://incompetech.com/..."}   # source_url optional

A track is only ever used if its license is in config music.allowed_licenses. The
credit line is appended to the post caption automatically.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .log import get_logger
from .state import JsonState

log = get_logger("music")

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".aiff"}
REQUIRED = ("title", "artist", "license", "credit")


@dataclass
class Track:
    path: Path
    title: str
    artist: str
    license: str
    credit: str
    source_url: str = ""
    placeholder: bool = False
    moods: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return self.path.name


def _norm_license(s: str) -> str:
    return " ".join(s.upper().replace("-", " ").split())


def sidecar_for(audio: Path) -> Path | None:
    for cand in (audio.with_suffix(".json"), audio.with_name(audio.name + ".json")):
        if cand.exists():
            return cand
    return None


def load_library(music_dir: Path, allowed_licenses: list[str]) -> tuple[list[Track], list[str]]:
    allowed = {_norm_license(x) for x in allowed_licenses}
    tracks, problems = [], []
    if not Path(music_dir).exists():
        return [], [f"{music_dir} does not exist"]
    for audio in sorted(Path(music_dir).iterdir()):
        if audio.suffix.lower() not in AUDIO_EXT or audio.name.startswith("."):
            continue
        side = sidecar_for(audio)
        if not side:
            problems.append(f"{audio.name}: no sidecar JSON (skipped)")
            continue
        try:
            meta = json.loads(side.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{side.name}: invalid JSON ({e})")
            continue
        missing = [k for k in REQUIRED if not str(meta.get(k, "")).strip()]
        if missing:
            problems.append(f"{side.name}: missing {', '.join(missing)} (skipped)")
            continue
        if _norm_license(meta["license"]) not in allowed:
            problems.append(f"{audio.name}: license '{meta['license']}' not in allowed_licenses (skipped)")
            continue
        tracks.append(Track(audio, meta["title"], meta["artist"], meta["license"], meta["credit"],
                            meta.get("source_url", ""), bool(meta.get("placeholder", False)),
                            tuple(meta.get("moods") or ())))
    return tracks, problems


def pick_track(tracks: list[Track], usage: JsonState | None = None, rng: random.Random | None = None) -> Track | None:
    """Least-used track first (ties broken randomly) so the library rotates evenly."""
    if not tracks:
        return None
    rng = rng or random.Random()
    counts = (usage.get("music", {}) if usage else {}) or {}
    low = min(counts.get(t.key, 0) for t in tracks)
    choice = rng.choice([t for t in tracks if counts.get(t.key, 0) == low])
    if usage is not None:
        counts[choice.key] = counts.get(choice.key, 0) + 1
        usage.set("music", counts)
    return choice
