"""ElevenLabs speech: the one place that talks to the ElevenLabs SDK.

Used by tts.ElevenLabsTTS (explainer narration, stories) and by voice.speak / speak_many (reels: originals, news reels, commentary) when
`voice.provider: elevenlabs`. A machine without ELEVENLABS_API_KEY (it lives in each machine's .env, never in the repo) or without a voice id is simply
"not available" and the callers fall back to the voices they had before.

Config (the `tts.elevenlabs` section): voice_id (the default voice), voices: {narrator, newsreader, commentator} (a voice id per role, optional),
model_id (default eleven_v4), stability, similarity_boost. Every call stitches to the lines around it (previous_text / next_text) so a video recorded line
by line still sounds like one take, asks for per-character timing (folded into words for captions and for cards timed to the words) and adds the characters
to data/state/elevenlabs_usage.json so the cost of a video can be seen.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path

from . import ffmpeg
from .log import get_logger
from .retry import PermanentError, call_with_retry

log = get_logger("elevenlabs")
DEFAULT_MODEL = "eleven_v4"
_LOCK = threading.Lock()


def voice_id_for(ecfg: dict, role: str = "narrator") -> str:
    return str(((ecfg.get("voices") or {}).get(role)) or ecfg.get("voice_id") or "").strip()


def available(ecfg: dict | None, role: str = "narrator") -> tuple[bool, str]:
    ecfg = ecfg or {}
    if not os.environ.get("ELEVENLABS_API_KEY"):
        return False, "ELEVENLABS_API_KEY not set on this machine (.env)"
    if not voice_id_for(ecfg, role):
        return False, f"no ElevenLabs voice chosen for {role!r} (tts.elevenlabs.voice_id)"
    try:
        import elevenlabs  # noqa: F401
    except ImportError:
        return False, "the elevenlabs package is not installed (pip install elevenlabs)"
    return True, "ok"


def client():
    from elevenlabs import ElevenLabs
    return ElevenLabs()                                        # reads ELEVENLABS_API_KEY from the environment


def _fold_words(alignment) -> list[dict]:
    """Per-character times -> words [{text, start, end}] (a space ends a word)."""
    g = (lambda k: alignment.get(k)) if isinstance(alignment, dict) else (lambda k: getattr(alignment, k, None))
    chars = g("characters") or []
    starts = g("character_start_times_seconds") or []
    ends = g("character_end_times_seconds") or []
    words, cur, start, end = [], "", None, 0.0
    for ch, s0, e0 in zip(chars, starts, ends):
        if str(ch).isspace():
            if cur:
                words.append({"text": cur, "start": round(start, 3), "end": round(end, 3)})
            cur, start = "", None
            continue
        if start is None:
            start = s0
        cur += str(ch)
        end = e0
    if cur:
        words.append({"text": cur, "start": round(start, 3), "end": round(end, 3)})
    return words


def _is_permanent(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    return code in (400, 401, 403, 404, 422)                    # bad key, bad voice, bad request: another try will not help


def synthesize(ecfg: dict, text: str, out_audio: Path, role: str = "narrator", previous_text: str | None = None, next_text: str | None = None,
               usage_file: Path | None = None) -> dict:
    """Say `text` -> `out_audio` (mp3). -> {"audio": Path, "words": [...], "characters": int}. Raises RuntimeError if ElevenLabs is not set up here."""
    ok, why = available(ecfg, role)
    if not ok:
        raise RuntimeError(why)
    from elevenlabs import VoiceSettings
    ecfg = ecfg or {}
    vs = VoiceSettings(stability=float(ecfg.get("stability", 0.5)), similarity_boost=float(ecfg.get("similarity_boost", 0.75)))
    kw = {"voice_id": voice_id_for(ecfg, role), "text": text, "model_id": ecfg.get("model_id") or DEFAULT_MODEL, "voice_settings": vs,
          "output_format": ecfg.get("output_format", "mp3_44100_128")}
    if previous_text:
        kw["previous_text"] = previous_text[-400:]
    if next_text:
        kw["next_text"] = next_text[:400]

    def once():
        try:
            return client().text_to_speech.convert_with_timestamps(**kw)
        except Exception as e:  # noqa: BLE001
            if _is_permanent(e):
                raise PermanentError(f"ElevenLabs refused the request ({getattr(e, 'status_code', '?')}): {e}") from e
            raise

    resp = call_with_retry(once, attempts=4, base=4, what="elevenlabs")
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    out_audio.write_bytes(base64.b64decode(getattr(resp, "audio_base_64", None) or resp.audio_base64))
    alignment = getattr(resp, "alignment", None) or getattr(resp, "normalized_alignment", None)
    words = _fold_words(alignment) if alignment else []
    _count(usage_file, len(text))
    log.info(f"elevenlabs ok: {len(text)} characters, {len(words)} words")
    return {"audio": out_audio, "words": words, "characters": len(text)}


def to_wav(audio: Path, wav: Path) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(audio), "-ar", "48000", "-ac", "1", str(wav)], timeout=120, what="elevenlabs to wav")
    return wav


def _count(usage_file: Path | None, characters: int) -> None:
    if not usage_file:
        return
    import json
    with _LOCK:
        try:
            data = json.loads(usage_file.read_text()) if usage_file.exists() else {}
        except Exception:  # noqa: BLE001
            data = {}
        month = time.strftime("%Y-%m")
        m = data.setdefault(month, {"characters": 0, "calls": 0})
        m["characters"] += int(characters)
        m["calls"] += 1
        usage_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = usage_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(usage_file)


def usage_file(root: Path) -> Path:
    return Path(root) / "data" / "state" / "elevenlabs_usage.json"


_BUDGET: list = [0.0, {}]


def budget(ttl: float = 600.0, warn_at: float = 0.8) -> dict:
    """This month's credits from the account: {used, limit, pct, resets (date), tier, warn}. Cached for `ttl` seconds; {} when ElevenLabs is not set up here
    or the account cannot be read. `warn` turns true at warn_at (80%) of the monthly limit, and a warning is logged once per reading."""
    if time.time() - _BUDGET[0] < ttl and _BUDGET[1]:
        return _BUDGET[1]
    if not os.environ.get("ELEVENLABS_API_KEY"):
        return {}
    try:
        s = client().user.subscription.get()
        used, limit = int(s.character_count), int(s.character_limit)
        out = {"used": used, "limit": limit, "pct": round(100 * used / limit, 1) if limit else 0.0, "tier": str(s.tier),
               "resets": time.strftime("%Y-%m-%d", time.localtime(s.next_character_count_reset_unix)) if s.next_character_count_reset_unix else "",
               "warn": bool(limit and used / limit >= warn_at)}
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read the ElevenLabs credits: {e}")
        return {}
    if out["warn"]:
        log.warning(f"ElevenLabs credits at {out['pct']}% of the month ({out['used']} of {out['limit']}); resets {out['resets']}")
    _BUDGET[0], _BUDGET[1] = time.time(), out
    return out
