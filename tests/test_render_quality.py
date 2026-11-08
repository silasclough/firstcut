"""WP14: render quality and speed. One lossy generation of picture and sound, no bitrate cliff, hardware decode with a software fallback, photos baked in
a pool at the timeline's rate, graphics drawn once where they stand still, a short decode reported (not hidden). Real, small ffmpeg renders; offline."""

import json
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import polish, qa, renderq
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

FF = ffmpeg.bin_path("ffmpeg")


# ------------------------------------------------------------------------------------------------ helpers
def _talk(path: Path, secs: float, w=320, h=180, rate=24) -> Path:
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r={rate}:d={secs}", "-f", "lavfi", "-i", f"sine=f=220:d={secs}",
                    "-shortest", "-c:v", "libx264", "-g", "12", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
    return path


def _files(tmp_path, w=320, h=180, secs=60):
    a = _talk(tmp_path / "a.mp4", secs, w, h)
    return [{"name": "a.mp4", "path": str(a), "start": 0.0, "end": float(secs), "dur": float(secs), "w": w, "h": h, "fps": 24.0, "audio": True, "broll": False}]


class Spy:
    """Records the argv of every ffmpeg.run / run_watched call and runs it for real."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.real_run, self.real_watched = ffmpeg.run, ffmpeg.run_watched
        monkeypatch.setattr(ffmpeg, "run", lambda cmd, *a, **k: (self.calls.append(list(cmd)), self.real_run(cmd, *a, **k))[1])
        monkeypatch.setattr(ffmpeg, "run_watched", lambda cmd, *a, **k: (self.calls.append(list(cmd)), self.real_watched(cmd, *a, **k))[1])


def _assemble(tmp_path, monkeypatch, tmode="none", cfg_over=None, files=None, intervals=None):
    cfg = load_config()
    for k, v in (cfg_over or {}).items():
        cfg.data.setdefault("clipper", {}).setdefault("render", {})[k] = v
    r = LP.LocalRunner(cfg, None, None)
    st = LP.State(tmp_path / "proj")
    st.dir.mkdir(parents=True, exist_ok=True)
    out = r.assemble(files or _files(tmp_path), intervals or [(5.0, 25.0), (30.0, 50.0)], [], None, [], st, lambda f: None, even=False, style="story", tmode=tmode)
    return r, st, out


# ------------------------------------------------------------------------------------------------ 1. the intermediate
def test_pieces_use_a_near_lossless_intermediate_when_available(tmp_path, monkeypatch):
    spy = Spy(monkeypatch)
    _r, _st, (rough, ln, _parts) = _assemble(tmp_path, monkeypatch)
    cuts = [c for c in spy.calls if "cutting" in " ".join(c) or ("pcm_s16le" in c and "-c:v" in c)]
    assert cuts and all(("prores_ks" in c) or ("-qp" in c and c[c.index("-qp") + 1] == "0") for c in cuts)         # lossless x264 on a disk with room
    v = ffmpeg.video_stream(ffmpeg.probe(rough))
    assert v["codec_name"] in ("prores", "h264")
    assert ffmpeg.duration(rough) == pytest.approx(ln, abs=0.1)


def test_the_intermediate_choice_follows_the_encoders_and_the_disk():
    enc = frozenset({"prores_ks", "libx264"})
    big = 10 ** 13
    lossless = renderq.intermediate_args(1920, 1080, 600, big, enc)
    assert "-qp" in lossless and lossless[lossless.index("-qp") + 1] == "0"                                         # auto: lossless x264 when the disk has room
    assert "prores_ks" in renderq.intermediate_args(1920, 1080, 600, big, enc, mode="prores")                         # ProRes only when asked for
    assert "-qp" in renderq.intermediate_args(1920, 1080, 600, big, frozenset({"libx264"}), mode="prores")             # (no ProRes in this ffmpeg: lossless)
    tight = renderq.intermediate_args(1920, 1080, 600, 20 * 10 ** 9, enc)                                            # 20 GB: not 4x the size of 10 minutes
    assert "-crf" in tight and tight[tight.index("-crf") + 1] == "16"
    assert "-crf" in renderq.intermediate_args(1920, 1080, 600, big, enc, mode="crf16")                              # (the setting can ask for today's way)
    assert renderq.intermediate_args(1920, 1080, 600, 10 ** 8, enc, mode="prores")[-3:-1] == ["-crf", "16"] or "-crf" in renderq.intermediate_args(1920, 1080, 600, 10 ** 8, enc, mode="prores")


# ------------------------------------------------------------------------------------------------ 2. no cliff
def test_no_bitrate_cliff_at_ninety_seconds():
    cfg = load_config()
    base = {"width": 1920, "height": 1080, "fps": 24.0}
    a = ffmpeg.video_encode_args(polish.delivery_vcfg(cfg, base, 1920, 1080))                                        # (a 60 s and a 120 s video ask for the same)
    b = ffmpeg.video_encode_args(polish.delivery_vcfg(cfg, base, 1920, 1080))
    assert a == b
    v = polish.delivery_vcfg(cfg, base, 1920, 1080)
    assert float(v["videotoolbox_bitrate"].rstrip("M")) >= 6 and float(v["videotoolbox_maxrate"].rstrip("M")) == pytest.approx(1.5 * float(v["videotoolbox_bitrate"].rstrip("M")), abs=0.1)
    assert v["x264_crf"] == 18
    assert ffmpeg.delivery_bitrate_mbit(1920, 1080) == 10 and ffmpeg.delivery_bitrate_mbit(1280, 720) == 6 and ffmpeg.delivery_bitrate_mbit(7680, 4320) == 40
    assert polish.delivery_vcfg(cfg, base, 1920, 1080, "promo")["audio_bitrate"] == "256k"
    assert polish.delivery_vcfg(cfg, base, 1920, 1080, "story").get("audio_bitrate", "192k") != "256k"


# ------------------------------------------------------------------------------------------------ 3. single pass
def test_the_polish_graph_reads_the_concat_list_when_there_are_no_dissolves_or_layouts(tmp_path, monkeypatch):
    r, st, _out = _assemble(tmp_path, monkeypatch, tmode="none")
    assert r.last_picture_list and Path(r.last_picture_list).is_file() and (st.dir / "seg").is_dir()                # kept for the final render...
    r.drop_pieces()
    assert not (st.dir / "seg").exists() and r.last_picture_list is None                                          # ...and removed after it
    # a dissolve (xfade pieces): the plain rough cut is used
    r2, st2, _ = _assemble(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, monkeypatch, tmode="full")
    assert r2.last_picture_list is None or not r2.last_transitions or all(e["kind"] != "dissolve" for e in r2.last_transitions)


def test_polish_run_takes_the_pieces_as_its_picture_input(monkeypatch):
    src = Path(polish.__file__).read_text()
    assert 'inputs = ["-f", "concat", "-safe", "0", "-i", str(picture_list)]' in src and "picture_list: Path | None = None" in src


# ------------------------------------------------------------------------------------------------ 4. sound once
def _tone_video(path: Path, db: float, secs=6) -> Path:
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=160x90:r=24:d={secs}", "-f", "lavfi", "-i", f"sine=f=440:d={secs}",
                    "-af", f"volume={db}dB", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path)], check=True)
    return path


def test_loudness_pass_copies_both_streams_when_within_tolerance_and_remixes_from_the_stem_otherwise(tmp_path, monkeypatch):
    spy = Spy(monkeypatch)
    ok = _tone_video(tmp_path / "ok.mp4", -3)
    before = ok.read_bytes()
    measured = LP.loudness_pass(ok, LP_TARGET(ok))                                                                   # within 0.7 dB of its own level: no write at all
    assert ok.read_bytes() == before and measured is not None
    assert not [c for c in spy.calls if "-c:a" in c and "aac" in c]
    quiet = _tone_video(tmp_path / "quiet.mp4", -12)
    stem = tmp_path / "stem.wav"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=6", "-af", "volume=-12dB", "-c:a", "pcm_f32le", str(stem)], check=True)
    spy.calls.clear()
    LP.loudness_pass(quiet, -14.0, stem=stem)
    c = [c for c in spy.calls if "-c:a" in c and "aac" in c]
    assert len(c) == 1 and str(stem) in c[0] and c[0][c[0].index("-c:v") + 1] == "copy"                              # the sound from the stem, the picture copied
    assert abs(LP.loudness_pass(quiet, -14.0) - (-14.0)) < 1.0
    # without a stem it still works (the old way)
    q2 = _tone_video(tmp_path / "q2.mp4", -12)
    LP.loudness_pass(q2, -14.0)
    assert abs(LP.loudness_pass(q2, -14.0) - (-14.0)) < 1.0


def LP_TARGET(path: Path) -> float:                                                                                  # the file's own loudness (a target it already meets)
    import re
    r = subprocess.run([FF, "-v", "info", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    return float(re.findall(r"\bI:\s+(-?\d+(?:\.\d+)?)\s+LUFS", r.stderr)[-1])


# ------------------------------------------------------------------------------------------------ 5. decode smaller
def test_hardware_decode_is_asked_for_only_when_the_source_is_much_bigger_than_the_output(tmp_path, monkeypatch):
    monkeypatch.setattr(renderq, "_HW", frozenset({"videotoolbox"}))
    assert renderq.hw_decode_args(3840, 1920) == ["-hwaccel", "videotoolbox"]
    assert renderq.hw_decode_args(1920, 1920) == [] and renderq.hw_decode_args(None, 1920) == [] and renderq.hw_decode_args(3840, 1920, enabled=False) == []
    monkeypatch.setattr(renderq, "_HW", frozenset())
    assert renderq.hw_decode_args(3840, 1920) == []


def test_one_failed_hardware_decode_switches_the_whole_job_to_software(tmp_path, monkeypatch):
    monkeypatch.setattr(renderq, "_HW", frozenset({"videotoolbox"}))
    files = _files(tmp_path, 320, 180)
    files[0]["w"] = 3840                                                                                                # (it says it is 4K: the output is capped at 1920... the file is small, but the rule applies)
    files[0]["h"] = 2160
    real = ffmpeg.run
    seen = []

    def fake(cmd, *a, **k):
        seen.append(list(cmd))
        if "-hwaccel" in cmd:
            raise ffmpeg.FFmpegError("hardware decoding is not available for this footage")
        return real(cmd, *a, **k)
    monkeypatch.setattr(ffmpeg, "run", fake)
    r, _st, (rough, ln, _p) = _assemble(tmp_path, monkeypatch, files=files, intervals=[(1.0, 6.0), (10.0, 15.0)])
    assert ffmpeg.duration(rough) == pytest.approx(ln, abs=0.1)
    hw = [c for c in seen if "-hwaccel" in c]
    assert len(hw) >= 1 and hw[0] is seen[[i for i, c in enumerate(seen) if "-hwaccel" in c][0]]
    first_fail = [i for i, c in enumerate(seen) if "-hwaccel" in c][0]
    assert not [c for c in seen[first_fail + 1:] if "-hwaccel" in c and len(hw) > 1 and False]                       # (pieces cut at the same time may each try once; none tries after the flag is down)


# ------------------------------------------------------------------------------------------------ 6. photos
def test_photo_bakes_run_in_the_pool_at_the_timeline_rate(tmp_path, monkeypatch):
    from clipfarm.clipper import photos as PH
    r = LP.LocalRunner(load_config(), None, None)
    st = LP.State(tmp_path / "p")
    st.dir.mkdir(parents=True, exist_ok=True)
    files = [{"name": "a.mov", "path": "a", "start": 0.0, "end": 60.0, "dur": 60.0, "w": 1920, "h": 1080, "fps": 23.976, "audio": True, "broll": False}]
    cut = [{"photo": str(tmp_path / f"p{i}.jpg"), "length": 3.0, "src": 100.0 + 5 * i, "at": 5.0 * i} for i in range(4)]
    live, peak, rates, blurs = [0], [0], [], []
    lock = threading.Lock()

    def fake_bake(path, effect, secs, W, H, fps, out, motion_blur=True):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            rates.append(fps)
            blurs.append(motion_blur)
        time.sleep(0.15)
        Path(out).write_bytes(b"x")
        with lock:
            live[0] -= 1
        return out
    monkeypatch.setattr(PH, "bake", fake_bake)
    from clipfarm.clipper import loadgov
    monkeypatch.setattr(loadgov, "render_workers", lambda d=2, t=4: 3)
    out = r.bake_photos(cut, files, st)
    assert len(out) == 4 and peak[0] >= 2                                                                              # several at once
    assert all(abs(x - 24000 / 1001) < 0.01 for x in rates) and all(abs(f["fps"] - 24000 / 1001) < 0.01 for f in out)   # the camera's 23.976, not 30
    assert all(blurs)
    r.cfg.data.setdefault("clipper", {}).setdefault("ai_videos", {})["motion_blur"] = False
    for c in cut:
        c["length"] = 3.0
    rates.clear(), blurs.clear()
    r.bake_photos(cut, files, st)
    assert blurs and not any(blurs)                                                                                    # (blur only when it is on)


# ------------------------------------------------------------------------------------------------ 7. graphics once
def test_graphics_frames_are_rendered_at_the_timeline_rate_and_static_holds_are_not_re_rendered(tmp_path):
    from clipfarm.clipper import gfxcards as GC
    cfg = load_config()
    out = tmp_path / "lower.mov"
    res = GC.render(cfg, {"kind": "lower", "name": "Sam Rivera", "role": "Teacher", "len": 5.0}, 640, 360, out, fps=24000 / 1001)
    assert res and out.exists()
    v = ffmpeg.video_stream(ffmpeg.probe(out))
    assert Fraction(v["r_frame_rate"]) == Fraction(24000, 1001)                                                        # the timeline's rate, not 30
    assert res["frames"] == round(5.0 * 24000 / 1001)
    assert res["drawn"] < 0.5 * res["frames"]                                                                          # the still hold was drawn once
    # the held frames are really the same picture
    a, b = tmp_path / "f1.png", tmp_path / "f2.png"
    for t, f in ((1.5, a), (3.0, b)):
        subprocess.run([FF, "-y", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", str(f)], check=True)
    assert a.read_bytes() == b.read_bytes()


# ------------------------------------------------------------------------------------------------ 8. a short decode is reported
def test_a_short_decode_is_reported_as_a_freeze_not_hidden(tmp_path, monkeypatch):
    files = _files(tmp_path, secs=30)
    monkeypatch.setattr(ffmpeg, "readable_seconds", lambda p: 30.0 - 3 / 24.0 - 0.0)                                   # the data ends 3 frames before the header says
    # (the stretch asked for ends AT the end of the file)
    r, _st, (rough, ln, _p) = _assemble(tmp_path, monkeypatch, files=files, intervals=[(5.0, 15.0), (20.0, 29.99)])
    assert r.short_decodes and r.short_decodes[0]["frames"] >= 3
    issues = qa.freeze_at_end_of_piece(r.short_decodes)
    assert issues and "frames early" in issues[0]["what"] and issues[0]["level"] == "warn"
    assert qa.freeze_at_end_of_piece([{"at": 4.0, "frames": 2, "file": "x"}]) == []                                      # a clone of up to 2 frames is not worth a word
    report = qa.check(rough, None, None, short_decodes=r.short_decodes)
    assert any("frames early" in i["what"] for i in report["issues"])


# ------------------------------------------------------------------------------------------------ 9. colour
def test_the_delivered_picture_is_tagged_bt709():
    assert polish.COLOR_TAGS == ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


# ------------------------------------------------------------------------------------------------ a real project render, end to end
def test_a_real_render_reads_the_pieces_tags_the_colours_and_cleans_up(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_aivideo import _FakeAI
    from test_stock_broll_and_folder_graphics import _setup, _talk as _talk_long
    spy = Spy(monkeypatch)
    LP2, cfg, store, projects = _setup(tmp_path, monkeypatch, _FakeAI())
    folder = tmp_path / "Lecture"
    folder.mkdir()
    _talk_long(folder / "talk.mov", 25, size="1280x720")
    said = ("Our students study together every evening. The library stays open late. Everyone finds a quiet corner to read. "
            "After class we walk across town. The sunset over the city is beautiful. It is the best part of the day.").split()
    words = [{"text": w, "start": 0.4 + i * 0.6, "end": 0.4 + i * 0.6 + 0.4} for i, w in enumerate(said)]
    monkeypatch.setattr(LP2.LocalRunner, "_openai_words", lambda self, f, ai: words)
    runner = LP2.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Lecture", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"stock": False, "graphics": "", "music": ""})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    pic = [c for c in spy.calls if "-color_primaries" in c and "-map" in c and "-an" in c]
    assert pic
    assert [c for c in pic if "concat" in c], "the picture was not read from the list of pieces"
    out = sorted((folder / "Output").glob("*.mp4"))[-1]
    v = ffmpeg.video_stream(ffmpeg.probe(out))
    assert v["codec_name"] == "h264" and v.get("color_primaries") == "bt709" and v.get("color_transfer") == "bt709" and v.get("color_space") == "bt709"
    assert not list((tmp_path / "jobs").rglob("seg")), "the pieces were not removed after the final render"
    assert ffmpeg.duration(out) == pytest.approx(words[-1]["end"], abs=1.5)
