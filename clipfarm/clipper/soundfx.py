"""Sound effects for the long-form explainers, made from scratch (no licensed audio): key clicks, taps, dings, whooshes, pops. (The card transition whooshes live in sfx.py.)

    cues_for(ai, concept, narration, words, dur) -> [{"t": seconds, "kind": "click|tap|ding|whoosh|pop", "gain": 0..1}]     (the AI places them to match the picture)
    render(cues, total_s, out_wav, level_db=-17) -> Path                                                                    (one stereo 48 kHz track, silent between cues)

The sounds are tiny synthesised waveforms (noise bursts, decaying sines, filtered sweeps), seeded so the same cue always sounds the same. They sit well under the voice.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000
KINDS = ("click", "tap", "ding", "whoosh", "pop")

CUE_SCHEMA = {"type": "object", "properties": {"cues": {"type": "array", "items": {"type": "object", "properties": {
    "t": {"type": "number"}, "kind": {"type": "string", "enum": list(KINDS)}, "gain": {"type": "number"}}, "required": ["t", "kind"]}}}, "required": ["cues"]}

CUE_SYSTEM = (
    "You are a sound designer for a YouTube explainer. You see ONE animated scene: what is drawn and how it moves (the concept), the narration spoken over it with the time "
    "each word is said, and its length. Place sound effects so they match what the viewer SEES, like a polished tech channel: "
    "'click' = one key press, for each character that is typed in a typing animation (space the clicks naturally, 0.06 to 0.12 s apart, a little irregular, at most 40 per scene); "
    "'tap' = a small thing lands or snaps into place (a tile, a label, a block); 'ding' = a result, answer or check mark appears, or something clicks into place correctly "
    "(at most 2); 'whoosh' = something slides, sweeps, dissolves or flies across (at most 3); 'pop' = something appears or bursts in (at most 4). "
    "Time each cue to the moment the thing happens, using the narration's word times as anchors (the scene's animation follows the words). "
    "Be tasteful: leave gaps, never more than about 12 effects in any 5 seconds apart from typing, no effect at all when nothing happens. "
    "gain 0.3 to 1 (1 = clear, 0.4 = subtle). Times are seconds from the start of the scene, between 0 and the length."
)


def cues_for(ai, concept: str, narration: str, words: list[dict], dur: float) -> list[dict]:
    timed = " ".join(f"{w['text']}@{float(w['start']):.1f}" for w in words if "start" in w)[:2000]
    prompt = f"SCENE (what is drawn and how it moves): {concept}\nNARRATION: {narration}\nTIMED WORDS: {timed}\nLENGTH: {dur:.2f} seconds\n"
    res = ai.complete_json("sfx_cues", CUE_SYSTEM, prompt, CUE_SCHEMA)
    out = []
    for c in res.data.get("cues", []):
        try:
            t, kind = float(c["t"]), str(c["kind"])
        except (KeyError, TypeError, ValueError):
            continue
        if kind in KINDS and 0.0 <= t <= dur:
            out.append({"t": round(t, 3), "kind": kind, "gain": max(0.2, min(1.0, float(c.get("gain", 0.7) or 0.7)))})
    return sorted(out, key=lambda c: c["t"])[:80]


# ------------------------------------------------------------------------------------------ the sounds
def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / max(decay, 1e-4))


def _bandpass_noise(rng, n: int, lo: float, hi: float) -> np.ndarray:
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(spec, n)
    return y / (np.max(np.abs(y)) + 1e-9)


def synth(kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "click":                                  # a soft key press: a short bright burst plus a small low body
        n = int(0.055 * SR)
        body = np.sin(2 * np.pi * rng.uniform(150, 190) * np.arange(n) / SR) * _env(n, 0.0004, 0.012)
        top = _bandpass_noise(rng, n, 1800, 5500) * _env(n, 0.0002, 0.006)
        return 0.55 * top + 0.35 * body
    if kind == "tap":                                    # a small thud
        n = int(0.14 * SR)
        t = np.arange(n) / SR
        f = 150 * np.exp(-t * 18) + 70
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.04) * 0.9 + 0.15 * _bandpass_noise(rng, n, 400, 1500) * _env(n, 0.0005, 0.01)
    if kind == "ding":                                   # a bell: two partials, a long soft decay
        n = int(0.9 * SR)
        t = np.arange(n) / SR
        y = np.sin(2 * np.pi * 1318.5 * t) * np.exp(-t / 0.22) + 0.5 * np.sin(2 * np.pi * 1975.5 * t) * np.exp(-t / 0.16) + 0.25 * np.sin(2 * np.pi * 2637 * t) * np.exp(-t / 0.08)
        return y * np.minimum(1.0, t / 0.003) * 0.6
    if kind == "whoosh":                                 # filtered noise that swells and fades
        n = int(0.5 * SR)
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        band = np.exp(-((np.log(np.maximum(f, 1)) - np.log(1800)) ** 2) / 0.9)
        y = np.fft.irfft(spec * band, n)
        env = np.sin(np.pi * np.clip(np.arange(n) / SR / 0.5, 0, 1)) ** 2
        return y / (np.max(np.abs(y)) + 1e-9) * env * 0.8
    if kind == "pop":                                    # a bubble: a short falling tone
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        f = 900 * np.exp(-t * 28) + 260
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.03) * 0.8
    raise ValueError(kind)


def render(cues: list[dict], total_s: float, out_wav: Path, level_db: float = -17.0) -> Path:
    """Mix the cues into one stereo track of `total_s` seconds. Each sound is peak-normalised, scaled by its gain, then the whole track is set to `level_db` peak."""
    n = int(total_s * SR) + SR
    track = np.zeros(n, dtype=np.float64)
    for k, c in enumerate(cues):
        s = synth(c["kind"], seed=k)
        s = s / (np.max(np.abs(s)) + 1e-9) * float(c.get("gain", 0.7))
        if c["kind"] == "click":
            s = s * rng_gain(k)                           # no two key presses exactly alike
        a = int(c["t"] * SR)
        if a >= n:
            continue
        b = min(n, a + len(s))
        track[a:b] += s[: b - a]
    peak = np.max(np.abs(track)) or 1.0
    track = track / peak * (10 ** (level_db / 20))
    track = track[: int(total_s * SR)]
    pan = np.stack([track, track], axis=1)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(pan, -1, 1) * 32767).astype(np.int16).tobytes())
    return out_wav


def rng_gain(k: int) -> float:
    return 0.75 + 0.25 * ((k * 2654435761) % 1000) / 1000.0
