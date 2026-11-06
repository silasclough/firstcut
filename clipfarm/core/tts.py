"""Text-to-speech providers behind one interface.

edge-tts (default): free Microsoft neural voices with word-boundary timestamps for captions.
kokoro: free local open-weights voice (work Mac, arm64). elevenlabs / openai: paid, optional.
macOS `say`: offline last resort (robotic). Providers without word timestamps get caption
timing from Whisper alignment (or estimation if Whisper is unavailable).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg
from .log import get_logger
from .retry import call_with_retry

log = get_logger("tts")


@dataclass
class TTSResult:
    audio: Path
    words: list[dict]            # [{"text", "start", "end"}] seconds
    duration: float
    provider: str
    voice: str
    meta: dict = field(default_factory=dict)


class TTSUnavailable(RuntimeError):
    pass


def _norm(tok: str) -> str:
    return re.sub(r"[^\w']", "", tok.lower().replace("’", "'"))


def align_to_text(text: str, words: list[dict]) -> list[dict]:
    """Give timed words the punctuation/casing of the original script.

    Boundary/Whisper words usually drop punctuation; captions look better with it and the
    caption grouper uses sentence ends. Unmatched words are kept as-is.
    """
    tokens = text.split()
    out, ti = [], 0
    for w in words:
        target = _norm(w["text"])
        match = None
        for look in range(ti, min(ti + 4, len(tokens))):
            if _norm(tokens[look]) == target and target:
                match = look
                break
        if match is not None:
            out.append({**w, "text": tokens[match]})
            ti = match + 1
        else:
            out.append(dict(w))
    return out


def estimate_word_timings(text: str, duration: float, lead: float = 0.05) -> list[dict]:
    """Spread words across `duration` weighted by length, with extra time after punctuation."""
    tokens = text.split()
    if not tokens:
        return []
    weights = []
    for t in tokens:
        w = max(2, len(_norm(t)))
        if re.search(r"[.!?]$", t):
            w += 6
        elif re.search(r"[,;:]$", t):
            w += 3
        weights.append(w)
    usable = max(0.1, duration - lead - 0.05)
    scale = usable / sum(weights)
    out, t = [], lead
    for tok, w in zip(tokens, weights):
        span = w * scale
        speak = span * (0.7 if re.search(r"[.!?,;:]$", tok) else 0.92)
        out.append({"text": tok, "start": round(t, 3), "end": round(t + speak, 3)})
        t += span
    return out


class TTSProvider(ABC):
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def available(self) -> tuple[bool, str]:
        return True, "ok"

    @abstractmethod
    def synthesize(self, text: str, out_dir: Path) -> TTSResult: ...


class EdgeTTS(TTSProvider):
    name = "edge"

    def available(self) -> tuple[bool, str]:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False, "edge-tts not installed"
        return True, "ok"

    def synthesize(self, text: str, out_dir: Path) -> TTSResult:
        import edge_tts

        voice = self.cfg.get("voice", "en-US-AndrewNeural")
        out_dir.mkdir(parents=True, exist_ok=True)
        audio = out_dir / "voice.mp3"
        tmp = out_dir / "voice.mp3.part"

        async def _run() -> list[dict]:
            words: list[dict] = []
            comm = edge_tts.Communicate(text, voice, rate=self.cfg.get("rate", "+0%"),
                                        pitch=self.cfg.get("pitch", "+0Hz"), boundary="WordBoundary")
            with open(tmp, "wb") as f:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        start = chunk["offset"] / 1e7
                        words.append({"text": chunk["text"], "start": round(start, 3),
                                      "end": round(start + chunk["duration"] / 1e7, 3)})
            return words

        words = call_with_retry(lambda: asyncio.run(_run()), attempts=3, base=3, what="edge-tts")
        if tmp.stat().st_size < 1000 or not words:
            raise TTSUnavailable("edge-tts returned no audio/word boundaries")
        tmp.replace(audio)
        return TTSResult(audio, align_to_text(text, words), ffmpeg.duration(audio), self.name, voice)


class MacSayTTS(TTSProvider):
    name = "say"

    def available(self) -> tuple[bool, str]:
        return (Path("/usr/bin/say").exists(), "ok" if Path("/usr/bin/say").exists() else "/usr/bin/say missing")

    def synthesize(self, text: str, out_dir: Path) -> TTSResult:
        voice = self.cfg.get("voice", "Eddy (English (US))")
        out_dir.mkdir(parents=True, exist_ok=True)
        aiff, audio = out_dir / "voice.aiff", out_dir / "voice.m4a"
        cmd = ["/usr/bin/say", "-r", str(self.cfg.get("rate_wpm", 190)), "-o", str(aiff)]
        if voice:
            cmd[1:1] = ["-v", voice]
        proc = subprocess.run(cmd + [text], capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise TTSUnavailable(f"say failed: {proc.stderr.decode(errors='replace')[:300]}")
        ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", aiff, "-c:a", "aac", "-b:a", "160k", audio], what="say->m4a")
        aiff.unlink(missing_ok=True)
        dur = ffmpeg.duration(audio)
        return TTSResult(audio, estimate_word_timings(text, dur), dur, self.name, voice, {"timing": "estimated"})


def whisper_timings(audio: Path, text: str, whisper_cfg: dict | None = None) -> tuple[list[dict], str]:
    """Word timings for audio that came without them: Whisper alignment, else estimation."""
    try:
        from . import whisper
        tr = whisper.transcribe(audio, whisper_cfg or {"backend": "auto", "model": "base.en"})
        words = whisper.all_words(tr)
        if words:
            return align_to_text(text, words), "whisper"
    except Exception as e:  # noqa: BLE001
        log.debug(f"whisper alignment unavailable: {e}")
    return estimate_word_timings(text, ffmpeg.duration(audio)), "estimated"


class ElevenLabsTTS(TTSProvider):
    """Premium voices through the ElevenLabs SDK (core/elevenlabs_voice.py): per-character timing folded into words, and each line stitched to the lines
    around it (previous_text / next_text) so a video recorded beat by beat still sounds like one take. Needs ELEVENLABS_API_KEY (.env) and
    tts.elevenlabs.voice_id; without them the chain moves on to the next provider."""
    name = "elevenlabs"

    def available(self) -> tuple[bool, str]:
        from . import elevenlabs_voice as EV
        return EV.available(self.cfg)

    def synthesize(self, text: str, out_dir: Path, previous_text: str | None = None, next_text: str | None = None) -> TTSResult:
        from . import elevenlabs_voice as EV
        out_dir.mkdir(parents=True, exist_ok=True)
        r = EV.synthesize(self.cfg, text, out_dir / "voice.mp3", previous_text=previous_text, next_text=next_text,
                          usage_file=EV.usage_file(Path(__file__).resolve().parents[2]))
        words = r["words"]
        if not words:
            words, _ = whisper_timings(r["audio"], text)
        return TTSResult(r["audio"], align_to_text(text, words), ffmpeg.duration(r["audio"]), self.name, EV.voice_id_for(self.cfg))


class OpenAITTS(TTSProvider):
    """OpenAI speech endpoint (audio only); caption timing from Whisper alignment.
    Needs OPENAI_API_KEY. (Written against the public API docs; not exercised without a key.)"""
    name = "openai"

    def available(self) -> tuple[bool, str]:
        import os
        return (True, "ok") if os.environ.get("OPENAI_API_KEY") else (False, "OPENAI_API_KEY not set")

    def synthesize(self, text: str, out_dir: Path) -> TTSResult:
        import os

        import requests
        body = {"model": self.cfg.get("model", "gpt-4o-mini-tts"), "voice": self.cfg.get("voice", "alloy"),
                "input": text, "response_format": "mp3"}
        if self.cfg.get("instructions"):
            body["instructions"] = self.cfg["instructions"]
        r = requests.post("https://api.openai.com/v1/audio/speech", json=body, timeout=180,
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        r.raise_for_status()
        out_dir.mkdir(parents=True, exist_ok=True)
        audio = out_dir / "voice.mp3"
        audio.write_bytes(r.content)
        words, how = whisper_timings(audio, text)
        return TTSResult(audio, words, ffmpeg.duration(audio), self.name, body["voice"], {"timing": how})


class KokoroTTS(TTSProvider):
    """Free, local, high-quality open-weights voice (Kokoro-82M, Apache-2.0). Needs `pip install kokoro
    soundfile` (torch; use the native arm64 Python on the work Mac). Timing via Whisper alignment."""
    name = "kokoro"
    _pipe = None

    def available(self) -> tuple[bool, str]:
        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"kokoro not installed ({type(e).__name__}); ./bootstrap.sh --with-ai on an arm64 Python"
        return True, "ok"

    def synthesize(self, text: str, out_dir: Path) -> TTSResult:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
        if KokoroTTS._pipe is None:
            KokoroTTS._pipe = KPipeline(lang_code=self.cfg.get("lang_code", "a"))
        voice = self.cfg.get("voice", "am_michael")
        chunks = [a for _, _, a in KokoroTTS._pipe(text, voice=voice, speed=float(self.cfg.get("speed", 1.08)))]
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / "voice.wav"
        sf.write(wav, np.concatenate([np.asarray(c) for c in chunks]), 24000)
        words, how = whisper_timings(wav, text)
        return TTSResult(wav, words, ffmpeg.duration(wav), self.name, voice, {"timing": how})


PROVIDERS: dict[str, type[TTSProvider]] = {
    "edge": EdgeTTS, "say": MacSayTTS, "elevenlabs": ElevenLabsTTS, "openai": OpenAITTS, "kokoro": KokoroTTS,
}


def cfg_for_category(tts_cfg: dict, category: str | None) -> dict:
    """Copy of the TTS config with edge.voice swapped for the category's voice, if mapped."""
    import copy
    out = copy.deepcopy(tts_cfg or {})
    voice = ((out.get("edge") or {}).get("voice_by_category") or {}).get(category or "")
    if voice:
        out.setdefault("edge", {})["voice"] = voice
    return out


def synthesize(text: str, tts_cfg: dict, out_dir: Path, dry_run: bool = False, previous_text: str | None = None, next_text: str | None = None) -> TTSResult:
    if dry_run:
        dur = max(1.0, len(text.split()) / 2.8)
        log.info(f"[dry-run] would synthesize {len(text.split())} words (~{dur:.0f}s)")
        return TTSResult(Path(out_dir) / "voice.mp3", estimate_word_timings(text, dur), dur, "dry-run", "-")
    errors = []
    for name in tts_cfg.get("provider_chain", ["edge", "say"]):
        cls = PROVIDERS.get(name)
        if not cls:
            errors.append(f"{name}: unknown provider")
            continue
        provider = cls(tts_cfg.get(name, {}))
        ok, why = provider.available()
        if not ok:
            errors.append(f"{name}: {why}")
            continue
        try:
            ctx = {"previous_text": previous_text, "next_text": next_text} if isinstance(provider, ElevenLabsTTS) else {}
            res = provider.synthesize(text, Path(out_dir), **ctx)
            log.info(f"tts ok via {name} ({res.duration:.1f}s, {len(res.words)} words)", extra={"provider": name})
            return res
        except Exception as e:  # noqa: BLE001 - fall through to the next provider
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log.warning(f"tts provider {name} failed: {e}")
    raise TTSUnavailable("all TTS providers failed: " + "; ".join(errors))


def change_tempo(res: TTSResult, factor: float, out_dir: Path) -> TTSResult:
    """Speed audio up by `factor` (pitch preserved) and rescale word timings."""
    if abs(factor - 1.0) < 1e-3:
        return res
    out = Path(out_dir) / f"voice_x{factor:.3f}.m4a"
    ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", res.audio, "-filter:a", f"atempo={factor:.4f}",
                "-c:a", "aac", "-b:a", "160k", out], what="atempo")
    words = [{**w, "start": round(w["start"] / factor, 3), "end": round(w["end"] / factor, 3)} for w in res.words]
    return TTSResult(out, words, ffmpeg.duration(out), res.provider, res.voice, {**res.meta, "tempo": factor})
