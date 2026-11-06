"""Optional: transcribe with OpenAI whisper-1 (word timestamps) instead of local faster-whisper.

Enabled with clipper.transcriber: openai (or auto + clipper.openai_whisper: true) and OPENAI_API_KEY
in .env. Audio is sent as 16 kHz mono MP3 in chunks well under the 25 MB upload limit.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import requests

from ..core import ffmpeg

CHUNK_S = 1200          # 20 min of 16 kHz mono @ 48 kbps ~ 7 MB
OVERLAP_S = 3.0         # each chunk hears this much either side of its share, so a word at a boundary is heard whole once


def has_punctuation(words: list[dict]) -> bool:
    """True if the word list carries sentence punctuation (Parakeet and re-punctuated whisper-1 do; old whisper-1 caches don't)."""
    return len(words) < 30 or any(str(w.get("text", "")).rstrip("\"')]")[-1:] in ".!?" for w in words)


def punctuate(words: list[dict], segments: list[dict]) -> list[dict]:
    """whisper-1's word list has no punctuation and no sentence capitals; the segment text of the same answer has both. Put them back
    onto the words, matched word for word in order (a word the two disagree on simply stays as it was)."""
    import difflib
    import re
    toks = [t for s in segments or [] for t in str(s.get("text", "")).split()]
    if not toks or not words:
        return words
    norm = lambda t: re.sub(r"[^\w']+", "", str(t).lower())          # noqa: E731
    a, b = [norm(w["text"]) for w in words], [norm(t) for t in toks]
    out = [dict(w) for w in words]
    for i, j, n in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(n):
            if a[i + k]:
                out[i + k]["text"] = toks[j + k]
    return out


def transcribe_mp3(mp3: Path, language: str = "en", key: str | None = None) -> list[dict]:
    """One MP3 chunk -> [{text, start, end}] (times relative to the chunk), with the punctuation of OpenAI's own segment text put back
    onto the words. Needs OPENAI_API_KEY (or the key you pass)."""
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    data = {"model": "whisper-1", "response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]}
    if language and language != "auto":
        data["language"] = language
    with Path(mp3).open("rb") as fh:
        r = requests.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                          files={"file": (Path(mp3).name, fh, "audio/mpeg")}, data=data, timeout=900)
    if r.status_code in (401, 403):
        raise RuntimeError(f"OpenAI rejected the API key ({r.status_code})")
    r.raise_for_status()
    body = r.json()
    words = [{"text": str(w.get("word", "")).strip(), "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)}
             for w in body.get("words", []) if str(w.get("word", "")).strip()]
    return punctuate(words, body.get("segments") or [])


def chunk_windows(total: float, chunk_s: float = CHUNK_S, overlap: float = OVERLAP_S) -> list[tuple[float, float, float, float]]:
    """[(audio_start, audio_len, keep_from, keep_to)] for transcribing `total` seconds in chunks. Each chunk's audio reaches `overlap`
    seconds into its neighbours, and only the words whose middle falls in its own share are kept: a word on a boundary is heard whole."""
    n = max(1, math.ceil(total / chunk_s))
    out = []
    for k in range(n):
        a, b = k * chunk_s, min(total, (k + 1) * chunk_s)
        s = max(0.0, a - (overlap if k else 0.0))
        e = min(total, b + (overlap if k + 1 < n else 0.0))
        out.append((s, e - s, a, b if k + 1 < n else float("inf")))
    return out


def keep_share(words: list[dict], offset: float, keep_from: float, keep_to: float) -> list[dict]:
    """A chunk's words moved to file time, only those whose middle is inside the chunk's own share."""
    out = []
    for w in words:
        s, e = round(w["start"] + offset, 3), round(w["end"] + offset, 3)
        if keep_from <= (s + e) / 2 < keep_to:
            out.append({"text": w["text"], "start": s, "end": e})
    return out


def transcribe_openai(src: Path, total: float, language: str, progress=None) -> list[dict]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    words: list[dict] = []
    wins = chunk_windows(total)
    with tempfile.TemporaryDirectory() as tmp:
        for k, (s, dur, keep_from, keep_to) in enumerate(wins):
            if progress:
                progress(k / len(wins))
            mp3 = Path(tmp) / f"c{k}.mp3"
            subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{s:.3f}", "-t", f"{dur:.3f}",
                            "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(mp3)],
                           check=True, timeout=900)
            words += keep_share(transcribe_mp3(mp3, language, key=key), s, keep_from, keep_to)
    if progress:
        progress(1.0)
    return sorted(words, key=lambda w: w["start"])
