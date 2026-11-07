"""Recap reels: the planning (speed, slow motion, windows, the reel's timeline) with no ffmpeg and no AI."""
from pathlib import Path

from clipfarm.clipper import recap as R


def words(n=120, step=0.5):
    return [{"text": f"w{i}" + ("." if i % 8 == 7 else ""), "start": i * step, "end": i * step + 0.4} for i in range(n)]


def test_speed_for_conforms_to_30():
    assert R.speed_for(60) == 0.5
    assert R.speed_for(59.94) == 0.5
    assert R.speed_for(50) == 0.6
    assert R.speed_for(120) == 0.25
    assert R.speed_for(240) == 0.25                      # never slower than quarter speed
    assert R.speed_for(30) == R.speed_for(24) == R.speed_for(29.97) == 1.0
    assert R.speed_for(60, slow=False) == 1.0


def test_three_in_four_fast_shots_are_slowed():
    got = [R.slow_this(60, k) for k in range(8)]
    assert sum(got) == 6 and not got[3] and not got[7]
    assert not R.slow_this(30, 0) and not R.slow_this(25, 5)
    assert R.slow_this(60, 0)                            # the first fast shot is slowed


def test_windows_mend_the_ai_and_never_overlap():
    w = words(200)
    got = R.windows(w, 100.0, [{"start": 0, "end": 30, "theme": "welcome"}, {"start": 20, "end": 50, "theme": "overlaps"}, {"start": 5, "end": 6}], 3, 30.0)
    assert len(got) == 3
    for x, y in zip(got, got[1:]):
        assert x["end"] <= y["start"] + 1e-6
    assert got[0]["theme"] == "welcome"
    assert all(g["end"] - g["start"] >= 5 for g in got)


def test_windows_without_the_ai():
    got = R.windows(words(), 60.0, None, 2, 25.0)
    assert len(got) == 2 and got[0]["end"] <= got[1]["start"] + 1e-6


def test_snap_does_not_cut_a_word():
    w = [{"text": "hello", "start": 1.0, "end": 2.0}]
    assert R.snap(w, 1.2, 5)[0] == 1.0
    assert R.snap(w, 1.9, 5)[0] == 2.0


def test_bites():
    assert R.bites(5) == [(0.0, R.BITE_S)]
    assert len(R.bites(12)) == 1 and len(R.bites(30)) == 2


def shots(n, fps=60, per=8.0):
    return [{"id": i, "file": f"f{i % 20}.mov", "path": f"/x/f{i % 20}.mov", "start": 0.0, "end": per, "what": f"s{i}", "quality": 6 + i % 3, "fps": fps} for i in range(n)]


def test_lay_out_fills_the_reel_exactly_and_slows_fast_shots():
    segs = R.merge_talk(R.lay_out(30.0, shots(40)))
    assert abs(sum(s["d"] for s in segs) - 30.0) < 0.02
    assert [s["kind"] for s in segs][0] == "talk" and segs[0]["d"] == R.BITE_S
    assert sum(1 for s in segs if s["kind"] == "talk") == 2     # the opening and one in the middle: "a little bit"
    b = [s for s in segs if s["kind"] == "broll"]
    assert all(1.0 <= s["d"] <= R.HOLD[1] + 1.01 for s in b[:-1])
    slow = [s for s in b if s["speed"] < 1]
    assert slow and all(s["speed"] == 0.5 for s in slow)
    assert 0.5 <= len(slow) / len(b) <= 0.9                      # most, not all
    for s in slow:
        assert abs(s["src_d"] - s["d"] * 0.5) < 0.01              # half as much footage for the same time on screen
    assert len({s["id"] for s in b}) == len(b)                    # no scene twice


def test_lay_out_normal_speed_footage_is_not_slowed():
    segs = R.lay_out(20.0, shots(20, fps=25))
    assert all(s.get("speed", 1.0) == 1.0 for s in segs)


def test_lay_out_limits_a_file_and_falls_back_to_the_speaker():
    one = [{"id": i, "file": "same.mov", "path": "/x/same.mov", "start": 0.0, "end": 9.0, "what": "a", "quality": 7, "fps": 30} for i in range(10)]
    segs = R.merge_talk(R.lay_out(30.0, one))
    assert sum(1 for s in segs if s["kind"] == "broll") == R.MAX_PER_FILE
    assert abs(sum(s["d"] for s in segs) - 30.0) < 0.02          # the speaker carries the rest; the reel is still the right length
    assert R.lay_out(10.0, [])[0]["kind"] == "talk"


def test_candidates_drop_poor_and_tiny_scenes():
    f = [{"name": "a", "path": "/a", "fps": 60}]
    sc = {"a": [{"start": 0, "end": 5, "what": "x", "quality": 8}, {"start": 5, "end": 9, "what": "y", "quality": 2}, {"start": 9, "end": 9.5, "what": "z", "quality": 9}]}
    c = R.candidates(f, sc)
    assert [x["what"] for x in c] == ["x"] and c[0]["fps"] == 60


def test_logo_and_broll_files(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "Acme Logo.png").write_bytes(b"x")
    (tmp_path / "b.mov").write_bytes(b"x")
    (tmp_path / ".hidden.mov").write_bytes(b"x")
    (tmp_path / "main.mp4").write_bytes(b"x")
    assert R.find_logo(tmp_path).name == "Acme Logo.png"
    assert R.find_logo(None) is None
    assert [p.name for p in R.broll_files(tmp_path, skip=tmp_path / "main.mp4")] == ["b.mov"]


# ---------------------------------------------------------------- a real render (ffmpeg), AI faked
def _ff(args):
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, timeout=120)


def test_a_reel_renders_vertical_30fps_with_slow_motion(tmp_path: Path):
    import json
    import subprocess

    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    main = tmp_path / "main.mp4"                                             # the "event video": 16:9 with talk-like sound
    _ff(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=14", "-f", "lavfi", "-i", "sine=f=300:d=14", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(main)])
    b60, b24 = tmp_path / "b60.mp4", tmp_path / "b24.mp4"
    _ff(["-f", "lavfi", "-i", "testsrc=s=960x540:r=60:d=8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(b60)])
    _ff(["-f", "lavfi", "-i", "mandelbrot=s=640x480:r=24", "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(b24)])
    cfg = load_config()
    eng = Engine(cfg, Store(tmp_path / "c"))
    job = eng.store.create({"app": "studio"}, {"mode": "recap", "caption_style": "bold", "music": "none"}, status="queued")
    ws = [{"text": f"word{i}", "start": 1.0 + i * 0.5, "end": 1.4 + i * 0.5} for i in range(20)]
    win = {"start": 1.0, "end": 13.0, "theme": "The day"}
    cands = R.candidates([{"name": "a", "path": str(b60), "fps": 60.0}, {"name": "b", "path": str(b24), "fps": 24.0}],
                         {"a": [{"start": 0, "end": 8, "what": "fast", "quality": 8}], "b": [{"start": 0, "end": 6, "what": "normal", "quality": 7}]})
    segs = R.merge_talk(R.lay_out(12.0, cands))
    assert any(s["kind"] == "broll" and s["speed"] == 0.5 for s in segs)
    clip = R.render_reel(eng, job["id"], 1, main, True, win, segs, ws, "The day", None, None, 0.0, {"caption_style": "bold"})
    out = eng.store.dir(job["id"]) / "clip_01.mp4"
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out)], capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["r_frame_rate"] == "30/1"
    assert abs(float(info["format"]["duration"]) - 12.0) < 0.3
    assert any(s["codec_type"] == "audio" for s in info["streams"])          # the talk plays underneath
    assert clip["slow_motion_shots"] >= 1 and (eng.store.dir(job["id"]) / "clip_01.srt").exists()


# ---------------------------------------------------------------- the app's side: starting a recap job
from test_app_portable import api, home  # noqa: E402,F401  (the same stand-in StudioAPI)


def test_a_recap_job_is_a_short_job_with_recap_options(api, tmp_path):
    import pytest
    src = tmp_path / "event.mp4"
    src.write_bytes(b"x")
    folder = tmp_path / "broll"
    folder.mkdir()
    jid = api.make_short({"recap": True, "source": str(src), "broll_folder": str(folder), "options": {"reels": 4, "reel_len": 45, "music": "none"}})["id"]
    o = api.store.get(jid)["options"]
    assert (o["mode"], o["recap_reels"], o["recap_len"], o["broll_dir"], o["music"]) == ("recap", 4, 45.0, str(folder), "none")
    assert api.store.get(jid)["title"].endswith("recap reels")
    o2 = api.store.get(api.make_short({"recap": True, "source": str(src)})["id"])["options"]
    assert (o2["recap_reels"], o2["recap_len"], o2["broll_dir"], o2["music"]) == (3, 30.0, "", "auto")        # the defaults: 3 reels of about 30 s, with music
    with pytest.raises(ValueError):
        api.make_short({"recap": True, "source": "https://example.com/v"})                                      # a link is not a finished video on this Mac
    with pytest.raises(ValueError):
        api.make_short({"recap": True, "source": str(src), "broll_folder": str(tmp_path / "nope")})
