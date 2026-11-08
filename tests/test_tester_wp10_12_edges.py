"""Tester's edge cases for WP10 (typed cards, scripture), WP12 (sound mix) and the eval-found fixes. Offline, tiny real ffmpeg only."""
import os
import signal
import stat
import subprocess
import time
from types import SimpleNamespace as N

import numpy as np
import pytest

from clipfarm.clipper import constraints as CO
from clipfarm.clipper import gfxcontent as G
from clipfarm.clipper import poolrules as PR
from clipfarm.clipper import scripture as SC
from clipfarm.clipper import soundmix as SM
from clipfarm.clipper import tighten as TI
from clipfarm.core import ffmpeg


def _say(text):
    ws, t = [], 0.0
    for w in text.split():
        ws.append({"text": w, "start": t, "end": t + 0.3})
        t += 0.35
    return ws, [N(start=0.0, end=t, text=text, idx=0)]


@pytest.mark.parametrize("text", ["I read 2 John last night and it was short", "John 5 times he said no", "Mark 3 of them came", "call 555-123-4567 now",
                                  "In 1999 and 2024 we went", "John 0:0 says hello", "Romans 999 says hi", "He said John 3:16-15 says x", "2 Romans says hi there"])
def test_scripture_does_not_fire_on_look_alikes(text):
    ws, s = _say(text)
    assert SC.find(ws, s) == []


@pytest.mark.parametrize("text,ref", [("Romans eight twenty-eight says all things work together", "Romans 8:28"),
                                      ("In 1 Corinthians 13:4-7 it says love is patient and kind and does not envy or boast.", "1 Corinthians 13:4-7"),
                                      ("First Corinthians thirteen says love is patient", "1 Corinthians 13"),
                                      ("Ps. 119:176 says I have gone astray like a lost sheep", "Psalms 119:176"),
                                      ("1 John 4:8 says God is love", "1 John 4:8")])
def test_scripture_forms(text, ref):
    ws, s = _say(text)
    assert [r["ref"] for r in SC.find(ws, s)] == [ref]


def test_empty_and_one_word_inputs_make_no_cards():
    assert SC.find([], []) == [] and G.candidates([], []) == []
    ws, s = _say("Hi")
    assert G.candidates(s, ws) == []


def test_numbers_money_percent_years():
    ws, s = _say("it cost $1.2 million dollars and rose 45% in 1999")
    shown = [c["item"]["value"] for c in G.candidates(s, ws) if c["type"] == "number"]
    assert "$1.2M" in shown and "45%" in shown and not any("1,999" in v or v == "1999" for v in shown)


def test_retime_at_the_last_second_never_negative_and_chapter_marks_are_filtered():
    c = G.retime({"kind": "stat", "value": "5", "label": "big deal", "at": 59.9}, None, out_len=60.0)
    assert c["len"] >= 0.0
    out = G.chapter_cards([{"at": 0.1, "title": "x", "n": 1}, {"at": 59.0, "title": "late", "n": 2}, {"at": 20, "title": "ok", "n": 3}, {"at": 25, "title": "near", "n": 4}], 60.0, hook_end=3.0)
    assert [m["text"] for m in out] == ["ok"]


@pytest.mark.parametrize("w,expect", [({"start_s": 1, "end_s": 3}, (1, 3)), ({"a": 1, "b": 3}, (1, 3)), ({"start": 0, "end": 2}, (0, 2)), ([1, 3], (1, 3)),
                                      ({"start_s": 5, "end_s": 3}, None), ([3, 3], None), ("abc", None), (None, None), ({}, None), ([1], None), ({"start_s": "x", "end_s": 3}, None),
                                      ({"start_s": float("nan"), "end_s": 3}, None), ({"start_s": 1}, None), ([float("inf"), float("inf")], None)])
def test_window_se_readers(w, expect):
    assert PR.window_se(w) == expect


def test_window_pairs_mixed_garbage_reversed_overlapping():
    wins = [{"start_s": 10, "end_s": 20}, [12, 30], "junk", None, [9, 2], {"a": 100, "b": 90}, 5]
    assert PR.window_pairs(wins) == [(10, 20), (12, 30)]
    assert PR.window_pairs(wins, lambda t: t * 2) == [(20, 40), (12, 30)]        # only dicts are mapped
    assert PR.window_pairs(wins, lambda t: None) == [(12, 30)]
    assert PR.window_pairs(None) == [] and PR.window_pairs("x") == [] and PR.window_pairs({"a": 1}) == []
    assert PR.rules_from({"exclude_windows": wins})["exclude_windows"] == [[10, 20], [12, 30]]
    kept, n = PR.outside_windows([{"at": 11, "length": 2}, {"at": 40, "length": 2}], wins)
    assert n == 1 and kept == [{"at": 40, "length": 2}]


def test_render_extra_includes_end_card_and_garbage_falls_back():
    plan = {"render_extra": {"end_card_s": 4.0}, "items": [{"kind": "talk"}, {"kind": "talk"}, {"kind": "cut"}]}
    assert CO.render_extra_s(plan) >= 4.0
    assert CO.render_extra_s({"render_extra": {"end_card_s": "x"}, "items": [{"kind": "talk"}]}) > 0
    assert CO.render_extra_s({"render_extra": "bad", "length_model": 5, "items": []}) == 0.0


def test_envelope_ducks_and_recovers_with_no_voice_file():
    words = [{"start": 5.0 + i * 0.3, "end": 5.0 + i * 0.3 + 0.25} for i in range(20)]   # speech 5-11 s
    pts = SM.envelope(None, 30.0, SM.DEFAULTS, words)
    assert abs(SM.gain_at(pts, 1.0)) < 0.2
    assert SM.gain_at(pts, 8.0) < -8.0
    assert SM.gain_at(pts, 29.0) > -0.5                                              # recovered by 29 s
    assert pts[-1][0] >= 30.0 - 0.01
    assert SM.envelope(None, 10.0, SM.DEFAULTS, [])[0][1] == 0.0                     # no speech: never ducked
    pts2 = SM.envelope(None, 10.0, SM.DEFAULTS, [{"start": 0.0, "end": 9.9}])         # speech at the very start and end
    assert SM.gain_at(pts2, 5.0) < -8.0
    assert SM.envelope(None, 0.0, SM.DEFAULTS, [{"start": "a"}, {"start": 1, "end": 0}, "x"])      # garbage words and a zero length do not raise


def test_envelope_of_a_two_hour_script_is_small_and_fast():
    words = [{"start": i * 2.0, "end": i * 2.0 + 1.5} for i in range(3600)]
    t0 = time.monotonic()
    pts = SM.envelope(None, 7200.0, SM.DEFAULTS, words)
    assert time.monotonic() - t0 < 20 and len(pts) < 20000


def test_style_music_bad_values_fall_back():
    sm = SM.style_music({"style_obj": {"music": {"attack_ms": "x", "release_ms": float("nan"), "bed_under_speech_lu": 0, "bed_in_gaps_lu": -10}}})
    assert sm["attack_ms"] == SM.DEFAULTS["attack_ms"] and sm["release_ms"] == SM.DEFAULTS["release_ms"] and sm["bed_under_speech_lu"] <= sm["bed_in_gaps_lu"]


def test_zero_cross_lands_on_a_crossing_and_missing_file_is_unchanged(tmp_path):
    wav = tmp_path / "s.wav"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=1:sample_rate=48000", str(wav)], check=True)
    t = TI.zero_cross(wav, 0.5013)
    assert abs(t - 0.5013) <= 0.0021
    x = np.frombuffer(subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{t - 0.001:.6f}", "-t", "0.002", "-i", str(wav), "-ac", "1", "-f", "f32le", "-"],
                                     capture_output=True).stdout, dtype="<f4")
    assert np.abs(x).min() < 0.02
    assert TI.zero_cross(tmp_path / "nope.wav", 1.234) == 1.234
    assert TI.zero_cross(wav, 99.0) == 99.0


def _fake(tmp_path, body):
    p = tmp_path / "ffmpeg"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def test_stop_killing_the_encoder_must_not_start_a_software_re_render(tmp_path, monkeypatch):
    """Stop SIGTERMs ffmpeg (localproj.hard_stop). encode_with_fallback must not treat that as 'hardware encoder failed' and re-run the whole encode."""
    calls = []

    def killed(cmd, stall_s=60.0, timeout=None, what="ffmpeg"):
        calls.append(cmd)
        raise ffmpeg.FFmpegError(f"{what} exited -15:\n")
    monkeypatch.setattr(ffmpeg, "run_watched", killed)
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.encode_with_fallback(lambda vc: ["ffmpeg"], {"encoder": "h264_videotoolbox"})
    assert len(calls) == 1, "Stop was followed by a software re-render"


def test_run_watched_sigterm_from_outside_ends_it_quickly(tmp_path):
    fake = _fake(tmp_path, "echo out_time_us=1\nsleep 30\n")
    import threading
    def killer():
        time.sleep(1.0)
        out = subprocess.run(["pgrep", "-f", str(tmp_path / "ffmpeg")], capture_output=True, text=True).stdout.split()
        for p in out:
            os.kill(int(p), signal.SIGTERM)
    threading.Thread(target=killer, daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.run_watched([fake, "-y", "x.mp4"], stall_s=60, what="the picture")
    assert time.monotonic() - t0 < 15


def test_fcpxml_with_a_real_long_envelope_validates_and_is_not_huge(tmp_path):
    """A 40-minute script's ducking keyframes (hundreds of them), a quiet aside, a song that must repeat: still valid FCPXML 1.11."""
    from fractions import Fraction
    from clipfarm.clipper import fcpxml as FX
    words = [{"start": i * 7.0, "end": i * 7.0 + 4.5} for i in range(340)]
    pts = SM.envelope(None, 2400.0, SM.DEFAULTS, words)
    kf = SM.final_cut_points(pts, -6.0)
    src = FX.Media("/Volumes/X/My Project/Media/source.mp4", 2500.0, fps=Fraction(30000, 1001))
    bed = FX.MusicBed("/Volumes/X/My Project/Music/a b.mp3", 200.0, "a b", start=10, keyframes=kf, crossfade=2.0)
    xml = FX.build("My Project", src, [(0.5, 2400.5)], 1920, 1080, Fraction(30000, 1001), music=bed)
    p = tmp_path / "p.fcpxml"
    p.write_text(xml)
    assert len(xml) < 5_000_000
    if not FX.DTD_DIR.exists():
        pytest.skip("Final Cut Pro not installed here")
    assert FX.validate(p) == []


def test_order_lines_on_a_20000_word_transcript_is_fast_and_garbage_items_are_skipped():
    from clipfarm.clipper import transcript_edit as TE
    words = [{"text": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(20000)]
    sents = [N(idx=k, start=words[k * 10]["start"], end=words[k * 10 + 9]["end"], text="x") for k in range(2000)]
    items = [{"kind": "talk", "a": k, "b": k} for k in range(1999, -1, -1)] + [{"kind": "talk", "a": 5, "b": 99999}, {"kind": "talk"}, {"kind": "cut"}, "junk"]
    items = [i for i in items if isinstance(i, dict)]
    plan = {"items": items, "te_cuts": [[100.0, 140.0]]}
    t0 = time.monotonic()
    lines = TE.order_lines(plan, sents, words)
    assert time.monotonic() - t0 < 10
    assert lines[0]["a"] == 19990 and all(l["b"] >= l["a"] for l in lines)
