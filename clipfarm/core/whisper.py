"""Speech-to-text with word timestamps: mlx-whisper (Apple Silicon native) or faster-whisper.

Audio is decoded with ffmpeg to 16 kHz mono float32 and passed in as a numpy array, so the
backends never need their own decoders.
"""

from __future__ import annotations

import functools
import platform
import subprocess
from pathlib import Path

import numpy as np

from . import ffmpeg
from .log import get_logger

log = get_logger("whisper")
SR = 16000


class WhisperUnavailable(RuntimeError):
    pass


def load_audio(path: Path | str, start: float | None = None, end: float | None = None, sr: int = SR) -> np.ndarray:
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None and start is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        raise ffmpeg.FFmpegError(proc.stderr.decode(errors="replace")[-500:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def pick_backend(cfg: dict) -> str:
    want = cfg.get("backend", "auto")
    if want != "auto":
        return want
    if platform.machine() == "arm64" and _have("mlx_whisper"):
        return "mlx-whisper"
    if _have("faster_whisper"):
        return "faster-whisper"
    raise WhisperUnavailable("no whisper backend installed (pip install faster-whisper, or mlx-whisper on arm64)")


@functools.lru_cache(maxsize=2)
def _fw_model(name: str, compute_type: str):
    from faster_whisper import WhisperModel

    log.info(f"loading faster-whisper model {name} ({compute_type}); first run downloads it")
    return WhisperModel(name, device="cpu", compute_type=compute_type)


def transcribe(path: Path | str, cfg: dict, start: float | None = None, end: float | None = None,
               language: str = "en") -> dict:
    """Returns {"backend", "model", "segments": [{start, end, text, words: [{text, start, end}]}]}.
    Times are relative to the original file (start offset added back)."""
    backend = pick_backend(cfg)
    audio = load_audio(path, start, end)
    offset = start or 0.0
    segments = []
    if backend == "faster-whisper":
        model_name = cfg.get("model", "base.en")
        model = _fw_model(model_name, cfg.get("compute_type", "int8"))
        segs, _info = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=False,
                                       beam_size=5, condition_on_previous_text=False)
        for s in segs:
            segments.append({
                "start": round(s.start + offset, 3), "end": round(s.end + offset, 3), "text": s.text.strip(),
                "words": [{"text": w.word.strip(), "start": round(w.start + offset, 3), "end": round(w.end + offset, 3)}
                          for w in (s.words or []) if w.word.strip()],
            })
    elif backend == "mlx-whisper":
        import mlx_whisper

        model_name = cfg.get("mlx_model", "mlx-community/whisper-base.en-mlx")
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=model_name, word_timestamps=True, language=language)
        for s in res.get("segments", []):
            segments.append({
                "start": round(s["start"] + offset, 3), "end": round(s["end"] + offset, 3), "text": s["text"].strip(),
                "words": [{"text": w["word"].strip(), "start": round(w["start"] + offset, 3),
                           "end": round(w["end"] + offset, 3)} for w in s.get("words", []) if w["word"].strip()],
            })
    else:
        raise WhisperUnavailable(f"unknown whisper backend {backend}")
    return {"backend": backend, "model": model_name, "segments": segments}


def all_words(transcript: dict) -> list[dict]:
    return [w for s in transcript.get("segments", []) for w in s.get("words", [])]
