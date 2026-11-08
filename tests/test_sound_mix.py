"""WP12: one ducking envelope for the MP4 and Final Cut, levels from the style, click-free joins, sparing effects, gains that reach Final Cut."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from clipfarm.clipper import fcpxml as FX
from clipfarm.clipper import music as MU
from clipfarm.clipper import polish as PL
from clipfarm.clipper import qa as QA
from clipfarm.clipper import sfx
from clipfarm.clipper import soundmix as SMX
from clipfarm.clipper import tighten as TI
from clipfarm.core import ffmpeg

FF = ffmpeg.bin_path("ffmpeg")
WORDS = [{"text": "one", "start": 2.0, "end": 2.6}, {"text": "two", "start": 2.7, "end": 4.0}, {"text": "three", "start": 8.0, "end": 9.5}]


def _pcm(path: Path, sr: int = 48000) -> np.ndarray:
    p = subprocess.run([FF, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-9)))


# ------------------------------------------------------------------------------------------------------------ the envelope
def test_the_envelope_ducks_inside_speech_with_the_style_attack_and_release():
    sm = SMX.style_music({})
    pts = SMX.envelope(None, 14.0, sm, WORDS)
    depth = sm["bed_under_speech_lu"] - sm["bed_in_gaps_lu"]                       # -10 dB
    assert SMX.gain_at(pts, 0.5) == pytest.approx(0.0, abs=0.2)                    # alone: the bed at its own level
    assert SMX.gain_at(pts, 3.0) == pytest.approx(depth, abs=0.3)                  # speaking: 10 dB further down (20 LU under the voice)
    assert SMX.gain_at(pts, 6.0) == pytest.approx(0.0, abs=0.3)                    # a gap of more than 1.5 s: back up
    assert SMX.gain_at(pts, 8.7) == pytest.approx(depth, abs=0.3)
    # attack 50 ms: most of the way down 150 ms after the padded start of the speech (1.85 s); release 400 ms: still well under 0 a time constant later
    start = 2.0 - SMX.PAD_S
    assert SMX.gain_at(pts, start + 0.15) < depth * 0.9
    end = 4.0 + SMX.PAD_S
    assert depth * 0.5 < SMX.gain_at(pts, end + 0.4) < 0.0
    assert SMX.gain_at(pts, end + 3.0) == pytest.approx(0.0, abs=0.5)
    assert len(pts) < 200                                                         # a thinned curve, not a point every 10 ms


def test_the_short_gaps_between_phrases_stay_ducked_and_a_quiet_aside_ducks_less(tmp_path):
    words = [{"text": "a", "start": 2.0, "end": 2.6}, {"text": "b", "start": 3.4, "end": 4.0}, {"text": "c", "start": 8.0, "end": 10.0}]
    pts = SMX.envelope(None, 14.0, SMX.style_music({}), words)
    assert SMX.gain_at(pts, 3.0) < -9.0                                           # 0.8 s between two words: the same speech, the music does not pump
    assert len(SMX.speech_spans(words)) == 2
    voice = tmp_path / "voice.wav"                                                # the first stretch loud, the second a quiet aside (26 dB down)
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=14:amplitude=0.5:seed=5", "-af",
                    "volume='if(between(t,2,4),1,if(between(t,8,10),0.05,0.0005))':eval=frame", "-ar", "48000", str(voice)], check=True, timeout=60)
    refined = SMX.envelope(voice, 14.0, SMX.style_music({}), words)
    assert SMX.gain_at(refined, 3.0) < -9.0                                       # the usual speech: the full duck
    assert -6.5 < SMX.gain_at(refined, 9.0) < -3.5                                # the quiet aside: about half of it


def test_the_sound_numbers_fall_back_to_defaults_for_nonsense():
    sm = SMX.style_music({"style_obj": {"music": {"voice_lufs": float("nan"), "attack_ms": "fast", "bed_under_speech_lu": -3, "bed_in_gaps_lu": -12}}})
    assert sm["voice_lufs"] == -16.0 and sm["attack_ms"] == 50.0
    assert sm["bed_under_speech_lu"] <= sm["bed_in_gaps_lu"]                      # (never louder under speech than alone)
    assert SMX.style_music(None) == SMX.style_music({"style_obj": "x"})


def test_the_mp4_bed_and_the_final_cut_keyframes_come_from_the_same_points(tmp_path):
    pts = SMX.envelope(None, 14.0, SMX.style_music({}), WORDS)
    wav = SMX.write_envelope_wav(pts, 14.0, tmp_path / "env.wav")
    ctrl = _pcm(wav, SMX.WAV_HZ)                                                    # what ffmpeg multiplies into the bed
    base = -4.5
    music = FX.MusicBed("/Volumes/X/Music/song.mp3", 40.0, "song", start=0.0, keyframes=SMX.final_cut_points(pts, base))
    clk = FX.Clock(Fraction(25))
    lines = FX._music_lines(clk, music, "r4", 0, 14 * 25)
    import re
    kf = [(float(Fraction(m.group(1).rstrip("s"))), float(m.group(2))) for m in re.finditer(r'<keyframe time="([^"]+)" value="(-?[\d.]+)dB"', "\n".join(lines))]
    assert len(kf) >= 6
    for t in np.linspace(0.5, 13.5, 10):
        mp4_db = 20 * np.log10(max(float(np.interp(t, np.arange(len(ctrl)) / SMX.WAV_HZ, ctrl)), 1e-6))
        fcp_db = float(np.interp(t, [a for a, _ in kf], [b for _, b in kf])) - base
        assert abs(mp4_db - fcp_db) < 0.5, (t, mp4_db, fcp_db)


# ------------------------------------------------------------------------------------------------------------ levels
def _gated_noise(path: Path, secs: float, spans: list[tuple[float, float]]) -> None:
    on = "+".join(f"between(t,{a},{b})" for a, b in spans)
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"anoisesrc=color=pink:duration={secs}:amplitude=0.5:seed=3",
                    "-af", f"volume='if({on},1,0.0005)':eval=frame,aformat=channel_layouts=stereo", "-ar", "48000", str(path)], check=True, timeout=120)


def test_voice_is_minus_sixteen_and_the_mix_minus_fourteen_within_one_lu_and_the_music_sits_under_the_voice(tmp_path):
    secs = 14.0
    spans = [(2.0, 4.0), (8.0, 9.5)]
    voice_src, music = tmp_path / "voice.wav", tmp_path / "music.wav"
    _gated_noise(voice_src, secs, spans)
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=330:duration={secs + 2}", "-af", "volume=0.5,aformat=channel_layouts=stereo",
                    str(music)], check=True, timeout=60)
    words = [{"text": "w", "start": a, "end": b} for a, b in spans]
    work = tmp_path / "work"
    work.mkdir()
    sm, env = SMX.style_music({}), []
    PL.make_sound(voice_src, [(0.0, secs)], False, True, "", True, music, 0.0, secs, -36.0, -14.0, work, work / "sound.m4a", {}, 300, sm=sm, words=words, env_out=env)
    assert env and env[0][0] == 0.0
    assert MU.integrated(work / "stem_voice.wav") == pytest.approx(-16.0, abs=1.0)           # the dialogue at -16 LUFS
    assert MU.integrated(work / "mix_final.wav") == pytest.approx(-14.0, abs=1.0)            # the finished mix at -14
    v, m = _pcm(work / "stem_voice.wav"), _pcm(work / "stem_music.wav")
    a, b = int(2.2 * 48000), int(3.8 * 48000)                                               # inside the first spoken stretch, past the attack
    gap = _rms_db(v[a:b]) - _rms_db(m[a:b])
    assert 18.0 <= gap <= 24.0, gap                                                         # 18 to 24 LU under the voice while it speaks
    alone = _rms_db(m[int(5.5 * 48000):int(7.0 * 48000)]) - _rms_db(m[a:b])
    assert alone == pytest.approx(10.0, abs=1.5)                                            # and about 10 dB louder when nobody speaks
    assert (work / "mix_final.wav").exists() and (work / "duck_env.wav").exists()


def test_without_words_the_old_compressor_still_makes_the_mix(tmp_path):
    voice_src, music = tmp_path / "voice.wav", tmp_path / "music.wav"
    _gated_noise(voice_src, 6.0, [(1.0, 3.0)])
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=8", "-af", "aformat=channel_layouts=stereo", str(music)], check=True)
    work = tmp_path / "w"
    work.mkdir()
    PL.make_sound(voice_src, [(0.0, 6.0)], False, True, "", True, music, 0.0, 6.0, -36.0, -14.0, work, work / "sound.m4a", {}, 300)
    assert MU.integrated(work / "mix_final.wav") == pytest.approx(-14.0, abs=1.0) and not (work / "duck_env.wav").exists()


def test_the_style_sets_the_level_where_the_bed_is_alone():
    assert MU.levels({}, WORDS, 10.0, SMX.style_music({})) == pytest.approx(-26.0)            # -16 voice, 10 LU under it
    assert MU.levels({}, WORDS, 10.0) in (-36.0, -26.0)                                       # (no style: the old numbers)


# ------------------------------------------------------------------------------------------------------------ joins
def test_cuts_land_on_zero_crossings_with_three_millisecond_fades_and_no_click(tmp_path):
    sine = tmp_path / "sine.wav"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-af", "volume=0.9", "-ar", "48000", str(sine)], check=True)
    t = 1.00037                                                                              # an arbitrary place in the wave
    z = TI.zero_cross(sine, t, 0.002)
    assert abs(z - t) <= 0.002 + 1e-4
    x = _pcm(sine)
    i = int(round(z * 48000))
    assert abs(x[i]) < 0.12                                                                  # on (next to) a zero crossing: 440 Hz at 0.9 changes ~0.05 per sample
    # a hard cut with 3 ms fades, as cut() makes them: the join has no jump
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    subprocess.run([FF, "-y", "-v", "error", "-i", str(sine), "-t", "0.8", "-af", "afade=t=out:st=0.797:d=0.003", str(a)], check=True)
    subprocess.run([FF, "-y", "-v", "error", "-ss", f"{z:.6f}", "-i", str(sine), "-t", "0.8", "-af", "afade=t=in:st=0:d=0.003", str(b)], check=True)
    joined = np.concatenate([_pcm(a), _pcm(b)])
    assert float(np.abs(np.diff(joined)).max()) < 0.05
    assert TI.zero_cross(tmp_path / "missing.wav", 1.0) == 1.0                              # no sound to read: the planned time stays


def test_zero_cross_stays_put_when_there_is_no_crossing(tmp_path):
    dc = tmp_path / "dc.wav"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=1", "-af", "volume=0.1,lowpass=f=30", "-ar", "48000", str(dc)], check=True)
    out = TI.zero_cross(dc, 0.5, 0.002)
    assert abs(out - 0.5) <= 0.0021


def test_a_click_at_a_cut_is_reported_by_qa_and_a_clean_join_is_not(tmp_path):
    sine = tmp_path / "sine.wav"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=200:duration=3", "-af", "volume=0.5", "-ar", "48000", str(sine)], check=True)
    # hard join at 1.0 s: the second part starts at the wave's peak (a 0.5 jump)
    a = tmp_path / "a.wav"
    subprocess.run([FF, "-y", "-v", "error", "-i", str(sine), "-t", "1.0", "-af", "afade=t=out:st=0.9:d=0.1,afade=t=in:st=0:d=0.001", str(a)], check=True)
    noisy = np.concatenate([_pcm(a), np.full(48000, 0.45), 0.45 * np.sin(2 * np.pi * 200 * np.arange(96000) / 48000)]).astype("<f4")
    raw = tmp_path / "hard.raw"
    raw.write_bytes(noisy.tobytes())
    hard = tmp_path / "hard.wav"
    subprocess.run([FF, "-y", "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(raw), str(hard)], check=True)
    found = QA.clicks(hard, [1.0])                                                           # the jump at 1.0 s (from ~0 to 0.45)
    assert found and found[0][0] == 1.0 and found[0][1] > 0.3
    smooth = tmp_path / "smooth.wav"
    subprocess.run([FF, "-y", "-v", "error", "-i", str(sine), "-af", "afade=t=out:st=0.99:d=0.003", str(smooth)], check=True)
    assert QA.clicks(smooth, [0.995, 2.0]) == []                                             # a pure tone has no click anywhere


def test_qa_check_reports_the_click_as_an_issue_with_its_code(tmp_path):
    x = np.concatenate([np.zeros(48000), np.full(48000, 0.8), np.zeros(48000)]).astype("<f4")
    raw = tmp_path / "c.raw"
    raw.write_bytes(x.tobytes())
    wav = tmp_path / "c.wav"
    subprocess.run([FF, "-y", "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(raw), str(wav)], check=True)
    mp4 = tmp_path / "c.mp4"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=3", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "pcm_s16le", "-shortest", str(mp4.with_suffix(".mov"))], check=True)
    rep = QA.check(mp4.with_suffix(".mov"), audio_cuts=[1.0, 2.0])
    codes = [i.get("code") for i in rep["issues"]]
    assert "click_at_cut" in codes
    assert not [i for i in QA.check(mp4.with_suffix(".mov"), audio_cuts=[0.5, 2.5])["issues"] if i.get("code") == "click_at_cut"]


# ------------------------------------------------------------------------------------------------------------ effects
def _cards(*times, style="wipe"):
    return [{"at": float(t), "len": 3.0, "style": style} for t in times]


def test_sfx_never_repeat_in_a_row_and_respect_the_style_count():
    ev = _cards(0, 40, 80, 120)                                                              # four cards a long way apart: all may sound
    out = sfx.choose(ev, {"max_per_30s": 2})
    assert len(out) == 4 and all(a["style"] != b["style"] for a, b in zip(out, out[1:]))     # (four wipes: the sound is rotated, never the same twice running)
    assert sfx.choose(ev, {"max_per_30s": 0}) == []                                          # a seminar / interview: no effects
    close = _cards(0, 8, 16, 24)                                                             # four cards in 30 s
    two = sfx.choose(close, {"max_per_30s": 2})
    assert len(two) <= 2 and len(sfx.choose(close, {"max_per_30s": 1})) == 1
    assert 0 < len(two) < len(close)                                                         # at least one stays silent
    both = sfx.choose(_cards(0, 10), {"max_per_30s": 5})
    assert len(both) == 1                                                                    # two cards in 30 s: one is silent even when many are allowed


def test_the_style_decides_whether_polish_writes_a_whoosh_track(tmp_path):
    ev = _cards(0, 40)
    assert sfx.track(ev, 60.0, tmp_path / "a.wav", policy={"max_per_30s": 0}) is None
    assert sfx.track(ev, 60.0, tmp_path / "b.wav", policy={"max_per_30s": 2}) is not None
    assert sfx.track(ev, 60.0, tmp_path / "c.wav") is not None                                # no policy: every card, as before


# ------------------------------------------------------------------------------------------------------------ Final Cut
def test_per_stretch_gains_reach_final_cut_as_adjust_volume(tmp_path):
    from clipfarm.clipper import localproj as LP
    base = {"kind": "clip", "path": "/Volumes/X/Shoot/C0001.MP4", "w": 1920, "h": 1080, "fps": 25.0, "audio": True, "file_dur": 600.0, "tc": 0.0, "tc_df": False}
    parts = [{"item": {**base, "start": 10.0}, "d": 4.0, "gain": -3.5, "a": {"clean_path": "/x/clean.wav"}},
             {"item": {**base, "start": 14.0}, "d": 4.0, "gain": -3.5, "a": {"clean_path": "/x/clean.wav"}},      # (next to it, same gain: one clip)
             {"item": {**base, "start": 18.0}, "d": 3.0, "a": {}}]                                                # (no gain: another clip)
    items = LP.fcp_items(parts)
    assert [(round(i["dur"], 1), i.get("gain_db"), bool(i.get("cleaned"))) for i in items] == [(8.0, -3.5, True), (3.0, None, False)]
    xml = FX.build_multicam("Talk", 1920, 1080, Fraction(25), None, items)
    assert 'adjust-volume amount="-3.5dB"' in xml and "voice cleaned in the MP4; raw here" in xml
    assert xml.count("adjust-volume") == 1                                                    # (the clip with no gain gets none)
    p = tmp_path / "g.fcpxml"
    p.write_text(xml)
    if FX.DTD_DIR.exists():
        assert FX.validate(p) == []


def test_a_multicam_stretch_carries_its_gain_before_the_angles():
    lines = FX._audio_lines({"gain_db": -2.25, "cleaned": True})
    assert lines[0].strip().startswith("<note>") and 'adjust-volume amount="-2.25dB"' in lines[1]
    assert FX._audio_lines({}) == []


# ------------------------------------------------------------------------------------------------------------ click check on real speech
def _say_wav(tmp_path, text):
    import shutil
    if not shutil.which("say"):
        pytest.skip("macOS say is not available")
    aiff = tmp_path / "s.aiff"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    wav = tmp_path / "s.wav"
    subprocess.run([FF, "-y", "-v", "error", "-i", str(aiff), "-ac", "1", "-ar", "48000", "-af", "volume=-6dB", str(wav)], check=True)
    return _pcm(wav)


def _wav_of(tmp_path, x, name):
    raw = tmp_path / (name + ".raw")
    raw.write_bytes(np.asarray(x, dtype="<f4").tobytes())
    wav = tmp_path / (name + ".wav")
    subprocess.run([FF, "-y", "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(raw), str(wav)], check=True)
    return wav


def test_a_hard_splice_of_real_speech_is_a_click_a_zero_crossing_join_and_a_drum_hit_are_not(tmp_path):
    sp = _say_wav(tmp_path, "The quick brown fox jumps over the lazy dog while the band plays on and on")
    sr = 48000
    loud = np.where(np.abs(sp) > 0.15)[0]
    a = int(loud[loud > sr][0])                                              # a loud sample after the first second
    b = next(int(i) for i in range(a + sr // 2, sp.size - 2000) if abs(sp[i] - sp[a]) > 0.12 and abs(sp[i]) > 0.1)
    hard = np.concatenate([sp[:a], sp[b:]])
    assert QA.clicks(_wav_of(tmp_path, hard, "hard"), [a / sr])                # the splice jumps: flagged
    gentle = lambda i: abs(sp[i]) < 0.01 and 0 < sp[i + 1] - sp[i] < 0.03     # a rising zero crossing on a gentle slope (what a join on a zero crossing looks for)
    za = next(int(i) for i in range(a, sp.size - 3) if gentle(i))
    zb = next(int(i) for i in range(b, sp.size - 3) if gentle(i))
    clean = np.concatenate([sp[:za + 1], sp[zb + 1:]])
    assert QA.clicks(_wav_of(tmp_path, clean, "clean"), [(za + 1) / sr]) == []   # a join on matching zero crossings is silent
    n = np.arange(sr)
    kick = np.concatenate([np.zeros(sr), 0.8 * np.sin(2 * np.pi * 60 * n / sr) * np.exp(-n / 8000.0), np.zeros(sr)])
    assert QA.clicks(_wav_of(tmp_path, kick, "kick"), [1.0]) == []             # a drum hit starting at the cut is not a click


def test_the_delivered_sound_has_its_true_peak_below_minus_one(tmp_path):
    """The limiter is a sample-peak limiter: true peaks got through (-0.6 dBTP measured). _finish_sound lowers the mix once so the AAC lands under -1."""
    sr = 48000
    n = np.arange(sr * 3)
    x = (0.98 * np.sin(2 * np.pi * 997.0 * n / sr) + 0.15 * np.sin(2 * np.pi * 11999.0 * n / sr)).clip(-0.999, 0.999)
    wav = _wav_of(tmp_path, x, "mix_final")
    afile = tmp_path / "out.m4a"

    def run(cmd, what):
        ffmpeg.run_watched(cmd, stall_s=60, what=what)
    before = QA.loud_and_peaks(wav)[1]
    assert before is not None and before > -1.0                                    # (the fixture really is over)
    PL._finish_sound(ffmpeg.bin_path("ffmpeg"), afile, 3.0, {}, run)
    assert QA.loud_and_peaks(afile)[1] <= -1.0
    assert QA.loud_and_peaks(tmp_path / "mix_final.wav")[1] <= -1.0
    quiet = _wav_of(tmp_path, 0.2 * np.sin(2 * np.pi * 440.0 * n / sr), "quiet")
    quiet.replace(tmp_path / "mix_final.wav")
    before = QA.loud_and_peaks(tmp_path / "mix_final.wav")[0]
    PL._finish_sound(ffmpeg.bin_path("ffmpeg"), afile, 3.0, {}, run)
    assert abs(QA.loud_and_peaks(tmp_path / "mix_final.wav")[0] - before) < 0.05    # a mix that is already under is left alone
