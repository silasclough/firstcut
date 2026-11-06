"""Word-timed transcripts with NVIDIA Parakeet (MLX, free, local; English). Used for AI-video projects: sharper word
times than the small Whisper model, which matters when cuts land between words.

Runs the `parakeet-mlx` command from its own venv (clipper.ai_videos.parakeet_bin, default ~/clipfarm/.venv-parakeet) so its
dependencies stay out of the main environment. The first run downloads the model (~2.4 GB).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.parakeet")
WHOLE_FILE_S = 600.0              # up to this long, a file is transcribed in one piece
DEFAULT_BIN = "~/clipfarm/.venv-parakeet/bin/parakeet-mlx"


def binary(cfg=None) -> Path | None:
    p = Path(str((cfg.get("clipper.ai_videos.parakeet_bin") if cfg else None) or DEFAULT_BIN)).expanduser()
    return p if p.exists() else None


def words_from_json(data: dict) -> list[dict]:
    """parakeet-mlx JSON (sentences -> sub-word tokens with start/end) -> [{text, start, end}] words."""
    words: list[dict] = []
    for s in data.get("sentences") or []:
        for t in s.get("tokens") or []:
            txt = str(t.get("text", ""))
            if not txt.strip():
                continue
            a, b = float(t["start"]), float(t.get("end", t["start"] + float(t.get("duration", 0))))
            if words and not txt.startswith(" "):               # a continuation piece of the previous word
                words[-1]["text"] += txt
                words[-1]["end"] = b
            else:
                words.append({"text": txt.strip(), "start": a, "end": b})
    return words


_GLUED = re.compile(r"^([A-Za-z][A-Za-z'\u2019]+)(\d[\d,.]*\d)([^\w]*)$")      # a word of 2+ letters, then 2+ digits ("was22", "it2019?")


_GLUED_SMALL = re.compile(r"^(by|of|to|at|in|on|for|from|than|about|over|under|was|is|are|and|or)(\d[\d,.]*)([^\w]*)$", re.I)     # "by8%": a short word, then even ONE digit


def split_glued_numbers(words: list[dict]) -> list[dict]:
    """Parakeet sometimes joins a number to the word before it ("was22", "it2019?"). A letters-then-digits token becomes two words, the time
    shared by their lengths. (Words with a digit inside such as "4K" or "COVID19" are left alone: they start with a capital/digit run.)"""
    out: list[dict] = []
    for w in words:
        m = _GLUED.match(str(w.get("text", ""))) or _GLUED_SMALL.match(str(w.get("text", "")))
        if not m or m.group(1).isupper():
            out.append(w)
            continue
        a, b, tail = m.group(1), m.group(2) + m.group(3), None
        total = len(a) + len(b)
        cut = round(w["start"] + (w["end"] - w["start"]) * len(a) / total, 3)
        out += [{**w, "text": a, "end": cut}, {**w, "text": b, "start": cut}]
    return out


def transcribe(src: Path, cfg=None, timeout: float = 4 * 3600) -> list[dict]:
    exe = binary(cfg)
    if exe is None:
        raise RuntimeError("parakeet-mlx isn't installed")
    with tempfile.TemporaryDirectory(prefix="parakeet-") as tmp:
        wav = Path(tmp) / "audio.wav"
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   what="audio for Parakeet")
        env = {**os.environ, "PATH": os.pathsep.join([str(Path(ffmpeg.bin_path("ffmpeg")).parent), "/opt/homebrew/bin",
                                                      "/usr/local/bin", os.environ.get("PATH", "")])}      # parakeet shells out to ffmpeg
        from . import net
        if not net.online():                                             # offline: the model on disk, no update check
            env["HF_HUB_OFFLINE"] = "1"
        dur = ffmpeg.duration(wav)

        def run(*chunking: str):
            return subprocess.run(["nice", "-n", "10", str(exe), str(wav), "--output-format", "json", "--output-dir", tmp, *chunking],
                                  capture_output=True, text=True, timeout=timeout, env=env)
        # Parakeet's default 120 s chunks (15 s overlap) lose words where the speech was cut and joined (an edited video: a quarter of the
        # words, whole sentences, were missing). A short file is transcribed WHOLE; a long one in 5-minute chunks with a wide overlap.
        out = Path(tmp) / "audio.json"
        r = run("--chunk-duration", "0") if dur <= WHOLE_FILE_S else run("--chunk-duration", "300", "--overlap-duration", "30")
        if (r.returncode or not out.exists()) and dur <= WHOLE_FILE_S:
            r = run("--chunk-duration", "300", "--overlap-duration", "30")
        if r.returncode or not out.exists():
            raise RuntimeError(f"parakeet failed: {(r.stderr or r.stdout)[-300:]}")
        words = split_glued_numbers(words_from_json(json.loads(out.read_text())))
    log.info(f"parakeet: {len(words)} words from {src.name}")
    from . import glossary
    return glossary.correct(words, glossary.terms(cfg))                # (Settings › How it edits › Names and words)
