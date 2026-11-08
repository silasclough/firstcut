"""One ducking envelope for the music bed, computed once and read by BOTH the finished video and the Final Cut project.

Before this, the MP4 ducked the music with a compressor (`sidechaincompress`) and Final Cut got a different, hand-made set of keyframes, so the two
mixes differed in level and shape. Now:

    envelope()          (time s, gain dB) points: 0 dB where the music is alone, `bed_under_speech_lu - bed_in_gaps_lu` (about -10 dB) while someone
                        speaks, with the style's attack and release as one-pole ramps; a quiet aside is ducked less (measured from the voice)
    write_envelope_wav  the same points as a control signal (a float wav) that ffmpeg multiplies into the bed (`amultiply`): sample-accurate, portable,
                        no filter-command syntax that differs between ffmpeg versions
    final_cut_points    the same points as volume keyframes for Final Cut, on top of the gain that sets the bed to the same level as in the MP4

Levels (dB are LU against the voice, which is at `voice_lufs`, -16): the bed sits `bed_in_gaps_lu` (-10) under the voice when it is alone and
`bed_under_speech_lu` (-20) under it while someone speaks (the 18-24 LU range sound mixers use).
"""

from __future__ import annotations

import math
import struct
import subprocess
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.soundmix")

DEFAULTS = {"voice_lufs": -16.0, "target_lufs": -14.0, "bed_under_speech_lu": -20.0, "bed_in_gaps_lu": -10.0, "attack_ms": 50.0, "release_ms": 400.0}
PAD_S = 0.15                         # speech spans are padded by this much each side
GAP_S = 1.5                          # a gap shorter than this between words stays ducked (the music does not pump between phrases)
QUIET_ASIDE_DB = 14.0                # a span this much under the usual speech level ducks the music only half as far
CONTROL_HZ = 100                     # the internal resolution of the envelope
WAV_HZ = 2000                        # the control wav's rate (ffmpeg resamples it to the audio rate)
SIMPLIFY_DB = 0.25                   # points are thinned until the curve is within this many dB of the full one


def _num(v, default: float, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(f) else max(lo, min(hi, f))


def style_music(opts: dict | None) -> dict:
    """The sound numbers of a job: the style's `music` section over DEFAULTS (bad values fall back to the default)."""
    m = (((opts or {}).get("style_obj") or {}).get("music") or {}) if isinstance((opts or {}).get("style_obj"), dict) else {}
    if not isinstance(m, dict):
        m = {}
    out = dict(DEFAULTS)
    out["voice_lufs"] = _num(m.get("voice_lufs"), out["voice_lufs"], -30.0, -6.0)
    out["target_lufs"] = _num(m.get("target_lufs"), out["target_lufs"], -24.0, -6.0)
    out["bed_under_speech_lu"] = _num(m.get("bed_under_speech_lu"), out["bed_under_speech_lu"], -40.0, -4.0)
    out["bed_in_gaps_lu"] = _num(m.get("bed_in_gaps_lu"), out["bed_in_gaps_lu"], -30.0, -2.0)
    out["attack_ms"] = _num(m.get("attack_ms"), out["attack_ms"], 5.0, 500.0)
    out["release_ms"] = _num(m.get("release_ms"), out["release_ms"], 50.0, 3000.0)
    if out["bed_under_speech_lu"] > out["bed_in_gaps_lu"]:                       # (under speech can never be louder than alone)
        out["bed_under_speech_lu"] = out["bed_in_gaps_lu"]
    return out


def bed_reference_lufs(sm: dict) -> float:
    """The loudness the bed is levelled to where it is alone (0 dB of the envelope): `bed_in_gaps_lu` under the voice."""
    return sm["voice_lufs"] + sm["bed_in_gaps_lu"]


def speech_spans(words: list[dict], pad: float = PAD_S, join_gap: float = GAP_S) -> list[tuple[float, float]]:
    """Where someone is speaking: the words' spans, padded, with the short gaps between phrases closed."""
    spans: list[list[float]] = []
    for w in sorted((w for w in words or [] if isinstance(w, dict)), key=lambda w: _num(w.get("start"), 0.0, -1e9, 1e9)):
        a, b = _num(w.get("start"), math.nan, -1e9, 1e9), _num(w.get("end"), math.nan, -1e9, 1e9)
        if not (math.isfinite(a) and math.isfinite(b)) or b < a:
            continue
        a, b = max(0.0, a - pad), b + pad
        if spans and a - spans[-1][1] <= join_gap:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans]


def _decode_mono(path: Path, sr: int = 8000, limit_s: float = 7200.0) -> np.ndarray:
    p = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-t", f"{limit_s:.0f}", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True, timeout=900)
    return np.frombuffer(p.stdout[: len(p.stdout) // 4 * 4], dtype="<f4").astype(np.float64)


def _span_levels(voice_wav: Path | None, spans: list[tuple[float, float]]) -> list[float | None]:
    """The median level (dB) of the voice inside each span (None when it can't be measured)."""
    if voice_wav is None:
        return [None] * len(spans)
    try:
        x = _decode_mono(Path(voice_wav))
    except Exception as e:  # noqa: BLE001 - the envelope then ducks every span the same
        log.debug(f"voice level for the envelope: {e}")
        return [None] * len(spans)
    if x.size < 800:
        return [None] * len(spans)
    sr, win = 8000, 800                                                        # 100 ms windows
    n = x.size // win
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    out: list[float | None] = []
    for a, b in spans:
        i, j = int(a * 10), max(int(a * 10) + 1, int(b * 10))
        seg = db[i:j]
        seg = seg[seg > -70]
        out.append(float(np.median(seg)) if seg.size >= 2 else None)
    return out


def _simplify(ts: np.ndarray, vs: np.ndarray, tol: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker: the fewest points whose straight lines stay within `tol` dB of the curve."""
    keep = np.zeros(len(ts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(ts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        line = vs[a] + (vs[b] - vs[a]) * (ts[a + 1:b] - ts[a]) / max(1e-9, ts[b] - ts[a])
        err = np.abs(vs[a + 1:b] - line)
        k = int(np.argmax(err))
        if err[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return [(round(float(t), 3), round(float(v), 2)) for t, v in zip(ts[keep], vs[keep])]


def envelope(voice_wav: Path | None, out_len: float, style: dict, words: list[dict]) -> list[tuple[float, float]]:
    """-> [(t s, gain dB)]: 0 dB where the music is alone, `bed_under_speech_lu - bed_in_gaps_lu` (about -10) under speech, shaped by the style's attack and
    release (one-pole ramps). The measured voice refines it: a span far quieter than the usual speech is ducked half as far."""
    sm = style if "attack_ms" in style and "bed_in_gaps_lu" in style else style_music({"style_obj": {"music": style}})
    out_len = max(0.1, _num(out_len, 0.1, 0.1, 86400.0))
    depth = sm["bed_under_speech_lu"] - sm["bed_in_gaps_lu"]                      # negative dB
    spans = [(a, min(b, out_len + 1.0)) for a, b in speech_spans(words) if a < out_len]
    levels = _span_levels(voice_wav, spans)
    known = [lv for lv in levels if lv is not None]
    usual = float(np.percentile(known, 75)) if known else None
    n = int(out_len * CONTROL_HZ) + 1
    target = np.zeros(n)
    for (a, b), lv in zip(spans, levels):
        d = depth * (0.5 if (lv is not None and usual is not None and lv < usual - QUIET_ASIDE_DB) else 1.0)
        i, j = int(a * CONTROL_HZ), min(n, int(math.ceil(b * CONTROL_HZ)) + 1)
        target[i:j] = np.minimum(target[i:j], d)
    dt = 1.0 / CONTROL_HZ
    k_att = 1 - math.exp(-dt / (sm["attack_ms"] / 1000.0))                         # going down (ducking): fast
    k_rel = 1 - math.exp(-dt / (sm["release_ms"] / 1000.0))                        # going up (the music returns): slow
    g = np.empty(n)
    cur = float(target[0])
    for i in range(n):
        tgt = target[i]
        cur += (k_att if tgt < cur else k_rel) * (tgt - cur)
        g[i] = cur
    ts = np.arange(n) * dt
    pts = _simplify(ts, g, SIMPLIFY_DB)
    if pts[-1][0] < out_len:
        pts.append((round(out_len, 3), pts[-1][1]))
    return pts


def gain_at(points: list[tuple[float, float]], t: float) -> float:
    """The envelope's value (dB) at time t (straight lines between the points)."""
    if not points:
        return 0.0
    ts = [p[0] for p in points]
    return float(np.interp(t, ts, [p[1] for p in points]))


def write_envelope_wav(points: list[tuple[float, float]], out_len: float, path: Path) -> Path:
    """The envelope as a mono float32 wav at WAV_HZ holding LINEAR gain: ffmpeg multiplies it into the bed (`amultiply`)."""
    n = int(max(0.1, out_len) * WAV_HZ) + 2
    t = np.arange(n) / WAV_HZ
    db = np.interp(t, [p[0] for p in points] or [0.0], [p[1] for p in points] or [0.0])
    lin = (10 ** (db / 20.0)).astype("<f4")
    data = lin.tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, WAV_HZ, WAV_HZ * 4, 4, 32) + b"data" + struct.pack("<I", len(data))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(hdr + data)
    return Path(path)


def final_cut_points(points: list[tuple[float, float]], base_db: float) -> list[tuple[float, float]]:
    """The envelope as Final Cut volume keyframes: the same points, on top of `base_db` (the gain that sets the song to the bed's level in the MP4)."""
    return [(t, round(base_db + v, 2)) for t, v in points]


def music_base_db(music_src, ref_lufs: float, start: float = 0.0, length: float | None = None) -> float:
    """The gain (dB) that brings the song to `ref_lufs`, as the MP4's loudnorm does; Final Cut's music clip carries it as its base level."""
    try:
        from . import music as MU
        lufs = MU.integrated(music_src)
    except Exception as e:  # noqa: BLE001
        log.debug(f"measuring the song for Final Cut: {e}")
        return 0.0
    if lufs is None or not math.isfinite(lufs):
        return 0.0
    return max(-30.0, min(12.0, ref_lufs - lufs))
