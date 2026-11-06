"""Studio-sound voice cleanup on this Mac (Descript calls it Studio Sound): DeepFilterNet takes the room noise, hum and echo out of
speech, so a weak recording can be used instead of only avoided (the workshop shoot's opening came from C0018's own camera mic across
the room: graded "bad" by soundcheck.py).

    available() -> bool                      the DeepFilterNet program is installed (Setup's "Voice cleanup" step)
    mode(kind, cfg) -> "auto"|"always"|"never" Settings › How it edits › Clean up voices (read fresh: no restart needed)
    wanted(grade, kind, cfg) -> bool         auto: only "weak" / "bad" sound (soundcheck grades); good sound is never touched
    cleaned_range(src, a, b, cache_dir, strength) -> (wav, starts_at)   a cleaned copy of seconds a..b of a recording (cached)
    clean(src_wav, dst_wav, strength)        the program itself, on one file

DeepFilterNet (MIT / Apache-2.0, github.com/Rikorose/DeepFilterNet) runs as its own small program (`deep-filter`, the release binary
for Apple silicon, model inside): no Python packages, works offline once Setup has downloaded it. Strength "light" caps the noise
reduction at 12 dB, so a voice never turns watery; "strong" at 24 dB. Only the MP4s FirstCut renders get the cleaned sound: Final
Cut always gets the original recordings (the cleaned copy is only ever read by the cutter, never named in the .fcpxml).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.voiceclean")

VERSION = "0.5.6"
URL = f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{VERSION}/deep-filter-{VERSION}-aarch64-apple-darwin"
SIZE = 27_877_081                         # the release file's exact size (no checksum is published; Setup also checks it is arm64 code)
SHA256 = "4601e7f4e4c03e59a4c5b5000216ef3add3e808799cfccd95e14e83ea4611081"   # (measured 2026-11-06 on the v0.5.6 release file)
MIN_GAIN_DB = 3.0                         # cleaning is kept only when the speech stands this much further above the room afterwards
MB = round(SIZE / 1e6)
ATTEN = {"light": 12, "strong": 24}       # dB: how much noise may be taken out at most
MARGIN = 1.0                              # seconds cleaned either side of a stretch (the filter needs a moment to settle)
MODES = ("auto", "always", "never")

_lock = threading.Lock()
_setting = {"at": 0.0, "val": {}}
_flags: dict = {}


class VoiceCleanError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ where it is, and whether to use it
def home() -> Path | None:
    h = os.environ.get("VIDEOBUILDER_HOME")
    return Path(h) if h else None


def binary(cfg=None) -> Path | None:
    """The deep-filter program: Setup's copy in <app home>/runtime/bin, a configured path, or one on PATH."""
    cands = []
    if cfg is not None:
        try:
            c = cfg.get("clipper.voice_clean_bin")
            if c:
                cands.append(Path(os.path.expanduser(str(c))))
        except Exception:  # noqa: BLE001
            pass
    if home():
        cands.append(home() / "runtime" / "bin" / "deep-filter")
    w = shutil.which("deep-filter")
    if w:
        cands.append(Path(w))
    return next((p for p in cands if p.is_file() and os.access(p, os.X_OK)), None)


def available(cfg=None) -> bool:
    return binary(cfg) is not None


def _app_settings() -> dict:
    """defaults.{long,short}.voices from the app's settings.json, read at most every 5 s (a change needs no restart)."""
    h = home()
    if not h:
        return {}
    if time.time() - _setting["at"] > 5:
        try:
            d = json.loads((h / "settings.json").read_text()).get("defaults") or {}
            v = {k: (d.get(k) or {}).get("voices") for k in ("long", "short")}
        except (OSError, ValueError, AttributeError):
            v = {}
        _setting.update(at=time.time(), val=v)
    return _setting["val"]


def mode(kind: str = "long", cfg=None) -> str:
    v = _app_settings().get(kind)
    if v is None and cfg is not None:
        try:
            v = cfg.get(f"clipper.voice_clean.{kind}") or cfg.get("clipper.voice_clean")
        except Exception:  # noqa: BLE001
            v = None
    return v if v in MODES else "auto"


def wanted(grade: str | None, kind: str = "long", cfg=None) -> bool:
    m = mode(kind, cfg)
    if m == "never" or not available(cfg):
        return False
    return m == "always" or grade in ("weak", "bad")


# ------------------------------------------------------------------------------------------------ the program
def _program_flags(exe: Path) -> dict:
    """Which option names this build of deep-filter takes (read once from its --help)."""
    key = str(exe)
    if key not in _flags:
        try:
            h = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=30)
            text = (h.stdout or "") + (h.stderr or "")
        except Exception:  # noqa: BLE001
            text = ""
        _flags[key] = {"atten": "--atten-lim-db" if "atten-lim" in text or not text else None,
                       "delay": "--compensate-delay" if "compensate-delay" in text or not text else None,
                       "out": "--output-dir" if "output-dir" in text or not text else "-o"}
    return _flags[key]


def clean(src_wav: Path, dst_wav: Path, strength: str = "light", cfg=None, run=subprocess.run) -> Path:
    """DeepFilterNet on one 48 kHz WAV -> dst_wav (written in one piece). VoiceCleanError when it can't."""
    exe = binary(cfg)
    if exe is None:
        raise VoiceCleanError("voice cleanup isn't installed (Settings › General › Run setup again)")
    f = _program_flags(exe)
    with tempfile.TemporaryDirectory(prefix="vclean-") as tmp:
        cmd = ["nice", "-n", "12", str(exe)]
        if f["delay"]:
            cmd.append(f["delay"])
        if f["atten"]:
            cmd += [f["atten"], str(ATTEN.get(strength, ATTEN["light"]))]
        cmd += [f["out"], tmp, str(src_wav)]
        r = run(cmd, capture_output=True, timeout=3600)
        got = Path(tmp) / Path(src_wav).name
        if getattr(r, "returncode", 1) or not got.exists() or got.stat().st_size < 1000:
            err = getattr(r, "stderr", b"") or b""
            raise VoiceCleanError(f"voice cleanup failed: {(err.decode(errors='replace') if isinstance(err, bytes) else err)[-200:]}")
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        part = dst_wav.with_name("." + dst_wav.name + ".part")
        shutil.move(str(got), part)
        part.replace(dst_wav)
    return dst_wav


def _gain(before: Path, after: Path) -> float | None:
    """How much further the speech stands above the room after cleaning (dB), or None when it can't be measured."""
    try:
        from . import soundcheck
        return float(soundcheck.measure(after)["snr"]) - float(soundcheck.measure(before)["snr"])
    except Exception:  # noqa: BLE001
        return None


def cleaned_range(src: str | Path, a: float, b: float, cache_dir: Path, strength: str = "light", cfg=None,
                  run=subprocess.run) -> tuple[Path, float]:
    """Seconds a..b of a recording (with MARGIN either side), cleaned: (the cleaned WAV, the source second it starts at). Cached by
    file, size, date, range, strength and version, so a re-render reuses it."""
    src = Path(src)
    st = src.stat()
    start = max(0.0, float(a) - MARGIN)
    end = float(b) + MARGIN
    key = hashlib.sha1(json.dumps([str(src), st.st_size, int(st.st_mtime), round(start, 2), round(end, 2), strength, VERSION]).encode()).hexdigest()[:16]
    dst = Path(cache_dir) / f"{key}.wav"
    if dst.exists() and dst.stat().st_size > 1000:
        return dst, start
    with _lock:                                                         # one at a time: it is a heavy job, and the cutter runs two
        if dst.exists() and dst.stat().st_size > 1000:
            return dst, start
        with tempfile.TemporaryDirectory(prefix="vclean-src-") as tmp:
            raw = Path(tmp) / f"{key}.wav"
            ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(raw)], what="sound for voice cleanup")
            t0 = time.monotonic()
            tmp_dst = dst.with_name(dst.name + ".part")
            clean(raw, tmp_dst, strength, cfg, run=run)
            # Kept only when it really helped. Measured on a workshop shoot C0018 (a camera mic across a hall): the voice went down with
            # the noise (speech over room 15.7 -> 16.1 dB): far-away speech can't be pulled out of its own echo, so it stays as it was.
            gain = _gain(raw, tmp_dst)
            if gain is not None and gain < MIN_GAIN_DB:
                shutil.copyfile(raw, tmp_dst)
                log.info(f"voice cleanup: {src.name} {start:.1f}-{end:.1f} s didn't help (+{gain:.1f} dB): kept as it was")
            else:
                log.info(f"voice cleanup: {src.name} {start:.1f}-{end:.1f} s in {time.monotonic() - t0:.1f} s"
                         + (f" (+{gain:.1f} dB)" if gain is not None else ""))
            os.replace(tmp_dst, dst)                                 # (only now can another caller see it)
    return dst, start
