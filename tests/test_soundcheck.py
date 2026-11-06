"""The sound check tells a close mic from a camera across the room (calibrated on Weekend Workshop, 2026-11-05: the opening came from
C0018's own mic, -44 dB speech 16 dB over the room; the recorder and the main cameras were -21 to -26 dB, 23-42 dB over it)."""
import numpy as np

from clipfarm.clipper import soundcheck as S


def _talk(level_db, room_db, seconds=20, sr=S.SR, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    room = rng.normal(0, 10 ** (room_db / 20), n)
    on = (np.sin(np.arange(n) / sr * 2 * np.pi * 0.7) > -0.2).astype(float)          # talking about 60% of the time
    voice = rng.normal(0, 10 ** (level_db / 20), n) * on
    return (room + voice).astype(np.float32)


def test_a_close_mic_is_good_and_a_far_quiet_one_is_bad():
    good = S.judge(_talk(-22, -60))
    assert good["grade"] == "good" and good["snr"] > 30
    bad = S.judge(_talk(-44, -58))
    assert bad["grade"] == "bad" and "very quiet" in bad["why"] and "far from the mic" in bad["why"]


def test_muffled_speech_is_caught_even_when_hiss_fills_the_high_frequencies():
    rng = np.random.default_rng(2)
    y = _talk(-24, -200)
    k = np.ones(64) / 64                                                              # a crude low-pass: the voice loses its consonants
    y = np.convolve(y, k, mode="same") + rng.normal(0, 10 ** (-52 / 20), len(y))      # ...and the room's hiss is broadband
    r = S.judge(y.astype(np.float32))
    assert "muffled" in r["why"] and r["grade"] != "good"


def test_silence_is_bad_not_an_error():
    assert S.judge(np.zeros(S.SR * 5, np.float32))["grade"] == "bad"
    assert S.judge(np.zeros(10, np.float32))["why"] == "no sound"
