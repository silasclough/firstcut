"""High-quality open-source text-to-speech for narration and commentary (no paid service).

Runs scripts/tts_worker.py in its own virtualenv (.venv-tts: mlx-audio on Apple Silicon) so the heavy dependencies stay
separate. Default engine: Kokoro (Apache-2.0, 82M parameters, natural, fast). Other open engines the worker can load:
Chatterbox (MIT), Qwen3-TTS (Apache-2.0). Config: voice.engine / voice.narrator / voice.commentator / voice.speed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .log import get_logger

log = get_logger("voice")


class VoiceUnavailable(RuntimeError):
    pass


def _paths(cfg):
    root = Path(cfg.root)
    py = root / ".venv-tts" / "bin" / "python"
    return py, root / "scripts" / "tts_worker.py"


_WARNED: set[str] = set()


def eleven_ready(cfg, role: str = "narrator") -> bool:
    """True when `voice.provider: elevenlabs` AND this machine can use it (key in .env, a voice chosen, the package installed). A machine that cannot says
    why once and keeps the open-source voices, so a missing key never stops a video."""
    if str((cfg.get("voice", {}) or {}).get("provider", "")).lower() != "elevenlabs":
        return False
    from . import elevenlabs_voice as EV
    ok, why = EV.available(cfg.get("tts.elevenlabs") or {}, role)
    if not ok and why not in _WARNED:
        _WARNED.add(why)
        log.warning(f"voice.provider is elevenlabs but {why}: using the local voice instead")
    return ok


def _eleven_lines(cfg, texts: list[str], outdir: Path, role: str) -> list[Path]:
    from . import elevenlabs_voice as EV
    ecfg = cfg.get("tts.elevenlabs") or {}
    usage = EV.usage_file(Path(cfg.root))
    outdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for i, t in enumerate(texts):
        try:
            r = EV.synthesize(ecfg, t, outdir / f"line_{i}.mp3", role=role, previous_text=texts[i - 1] if i else None,
                              next_text=texts[i + 1] if i + 1 < len(texts) else None, usage_file=usage)
            outs.append(EV.to_wav(r["audio"], outdir / f"line_{i}.wav"))
        except Exception as e:  # noqa: BLE001
            raise VoiceUnavailable(f"ElevenLabs failed on line {i}: {e}") from e
    return outs


def available(cfg) -> tuple[bool, str]:
    """Can this machine speak at all: ElevenLabs when it is the provider and set up, else the local open-source voice."""
    if eleven_ready(cfg, "narrator"):
        return True, "ElevenLabs"
    return local_available(cfg)


def local_available(cfg) -> tuple[bool, str]:
    py, worker = _paths(cfg)
    if not py.exists():
        return False, f"{py} missing (create it: python3.12 -m venv .venv-tts && .venv-tts/bin/pip install mlx-audio soundfile)"
    return (worker.exists(), "ok" if worker.exists() else f"{worker} missing")


FILLERS = {"huh", "uh", "um", "uhm", "hmm", "mm", "mmm", "ah", "oh", "er", "erm", "hm"}


def _tokens(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9']+", text.lower())


def heard(cfg, wav: Path) -> list[str]:
    """What the voice actually said (speech recognition on the recording)."""
    from . import whisper
    tr = whisper.transcribe(wav, cfg.get("captions.whisper") or {"backend": "auto", "model": "base.en"})
    return _tokens(" ".join(str(w.get("text", "")) for w in whisper.all_words(tr)))


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _cardinal(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            return _cardinal(n // size) + [name] + (_cardinal(n % size) if n % size else [])
    return []


def _number_forms(token: str) -> list[list[str]]:
    """How a digit string may be SAID: as a cardinal ("two hundred fifty nine"), a year ("twenty twenty six"), digit by digit."""
    if not token.isdigit() or len(token) > 12:
        return []
    n = int(token)
    forms = [_cardinal(n)]
    if 1100 <= n <= 2099:
        forms.append(_cardinal(n // 100) + (_cardinal(n % 100) if n % 100 >= 10 else (["oh"] + _cardinal(n % 100) if n % 100 else ["hundred"])))
    forms.append([_ONES[int(c)] for c in token])
    return forms


def _spell_digits(said: list[str], want: list[str]) -> list[str]:
    """Speech recognition writes numbers as digits ("259"), a script that is read aloud spells them: put the digits back into words, in the
    form the script used when it used one of the usual ones."""
    out, text = [], " " + " ".join(want) + " "
    for w in said:
        forms = _number_forms(w)
        if not forms:
            out.append(w)
            continue
        out += next((f for f in forms if " " + " ".join(f) + " " in text), forms[0])
    return out


def clean(said: list[str], text: str) -> tuple[bool, str]:
    """Did the voice read the line without inventing sounds ("huh, huh") or dropping words?"""
    want = _tokens(text)
    said = _spell_digits(said, want)
    want = _spell_digits(want, want)                      # digits in the script itself are read aloud too
    ws = set(want)
    fill = [w for w in said if w in FILLERS and w not in ws]
    extra = [w for w in said if w not in ws]
    missing = [w for w in want if w not in set(said)]
    if fill:
        return False, f"added filler sounds {fill[:4]}"
    if len(extra) > max(1, int(0.2 * len(want))):
        return False, f"added words {extra[:4]}"
    if len(missing) > max(1, int(0.25 * len(want))):
        return False, f"dropped words {missing[:4]}"
    return True, ""


def speak_checked(cfg, texts: list[str], outdir: Path, role: str = "narrator", tries: int = 3) -> list[Path]:
    """speak_many, then listen to every line: if the voice made up sounds ("huh") or dropped words, say that line again
    (slightly different wording / speed), and as a last resort in the lighter Kokoro voice."""
    outs = speak_many(cfg, texts, outdir, role=role)
    for i, (t, o) in enumerate(zip(texts, outs)):
        for attempt in range(tries):
            try:
                ok, why = clean(heard(cfg, o), t)
            except Exception as e:  # noqa: BLE001 - can't check: trust it
                log.warning(f"couldn't check voice line {i}: {e}")
                break
            if ok:
                break
            log.warning(f"voice line {i} was not clean ({why}); redoing (try {attempt + 1})")
            if attempt == tries - 1:
                try:
                    speak(cfg, t, o, role=role, engine="kokoro", voice="am_michael")
                except VoiceUnavailable as e:                                   # no local voice on this machine (ElevenLabs only): keep the line as it is
                    log.warning(f"voice line {i} stays as recorded: {e}")
            else:
                alt = t.replace("?", ".").replace("!", ".").replace("...", ".") if attempt == 0 else t
                speak(cfg, alt, o, role=role, speed=0.96 if attempt == 0 else 1.04)
    return outs


def speak_many(cfg, texts: list[str], outdir: Path, role: str = "commentator", timeout: float = 1800.0) -> list[Path]:
    """Several short lines in one go (the model loads once). -> one wav per line, in order."""
    import json
    if eleven_ready(cfg, role):
        try:
            return _eleven_lines(cfg, texts, outdir, role)
        except VoiceUnavailable as e:                                                       # credits ran out, a network problem: the local voice takes over if there is one
            if not local_available(cfg)[0]:
                raise
            log.warning(f"{e}; using the local voice for these lines")
    ok, why = local_available(cfg)
    if not ok:
        raise VoiceUnavailable(why)
    py, worker = _paths(cfg)
    vcfg = cfg.get("voice", {}) or {}
    outdir.mkdir(parents=True, exist_ok=True)
    outs = [outdir / f"line_{i}.wav" for i in range(len(texts))]
    batch = outdir / "batch.json"
    batch.write_text(json.dumps([{"text": t, "out": str(o)} for t, o in zip(texts, outs)]))
    cmd = [str(py), str(worker), "--engine", vcfg.get("engine", "kokoro"), "--voice",
           vcfg.get(role, "am_michael"), "--speed", str(vcfg.get("speed", 1.0)), "--batch", str(batch)]
    if vcfg.get(f"instruct_{role}"):
        cmd += ["--instruct", str(vcfg[f"instruct_{role}"])]                  # the delivery: calm and plain
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not all(o.exists() for o in outs):
        raise VoiceUnavailable(f"voice failed: {(proc.stderr or proc.stdout)[-400:]}")
    return outs


def speak(cfg, text: str, out: Path, role: str = "narrator", voice: str | None = None, engine: str | None = None,
          speed: float | None = None, timeout: float = 900.0, instruct: str | None = None) -> Path:
    """Say `text` -> `out` (wav). role: 'narrator' (the main voice of an original video) or 'commentator' (short asides)."""
    if engine is None and voice is None and eleven_ready(cfg, role):
        import shutil
        tmp = out.parent / f".{out.stem}_el"
        try:
            got = _eleven_lines(cfg, [text], tmp, role)[0]
            out.parent.mkdir(parents=True, exist_ok=True)
            got.replace(out)
            return out
        except VoiceUnavailable as e:
            if not local_available(cfg)[0]:
                raise
            log.warning(f"{e}; using the local voice for this line")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    ok, why = local_available(cfg)
    if not ok:
        raise VoiceUnavailable(why)
    py, worker = _paths(cfg)
    vcfg = cfg.get("voice", {}) or {}
    eng = engine or vcfg.get("engine", "kokoro")
    v = voice or vcfg.get(role, "am_michael" if role == "narrator" else "af_heart")
    sp = float(speed if speed is not None else vcfg.get("speed", 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), str(worker), "--engine", eng, "--voice", v, "--speed", str(sp), "--text", text, "--out", str(out)]
    if instruct is not None:
        if instruct:
            cmd += ["--instruct", instruct]
    elif vcfg.get(f"instruct_{role}") and eng == vcfg.get("engine"):
        cmd += ["--instruct", str(vcfg[f"instruct_{role}"])]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out.exists():
        raise VoiceUnavailable(f"{eng} failed: {(proc.stderr or proc.stdout)[-400:]}")
    return out
