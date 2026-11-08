"""Phase 1: a messy folder (mixed fps/containers, a corrupt file, a silent file, a 0-byte file, a 3.5 s clip) made into a video, then the saved plan
loses its Phase 1 fields (sid, style_obj, file seconds) and is rendered again. The delivered file is probed."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_aivideo import _FakeAI  # noqa: E402
from test_stock_broll_and_folder_graphics import _setup, _talk  # noqa: E402
from clipfarm.core import ffmpeg  # noqa: E402

FF = ffmpeg.bin_path("ffmpeg")


def _silent(path, secs, size="320x180", rate=24, fmt=None):
    cmd = [FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={secs}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    subprocess.run(cmd + (["-f", fmt] if fmt else []) + [str(path)], check=True, timeout=120)


@pytest.mark.parametrize("size", ["1280x720", "720x1280"])
def test_messy_folder_renders_and_an_old_plan_renders_again(tmp_path, monkeypatch, size):
    LP2, cfg, store, projects = _setup(tmp_path, monkeypatch, _FakeAI())
    folder = tmp_path / "Messy"
    folder.mkdir()
    _talk(folder / "talk.mov", 25, size=size)
    _silent(folder / "C0020.MP4", 3.5, rate=30)
    _silent(folder / "clip.mts", 4, size="640x360", rate=25, fmt="mpegts")
    _silent(folder / "noaudio.mov", 5, rate=60)
    (folder / "corrupt.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"garbage" * 50)
    (folder / "empty.mov").write_bytes(b"")
    said = ("Our students study together every evening. The library stays open late. Everyone finds a quiet corner to read. "
            "After class we walk across town. The sunset over the city is beautiful. It is the best part of the day.").split()
    words = [{"text": w, "start": 0.4 + i * 0.6, "end": 0.4 + i * 0.6 + 0.4} for i, w in enumerate(said)]
    monkeypatch.setattr(LP2.LocalRunner, "_openai_words", lambda self, f, ai: words)
    runner = LP2.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Messy", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"stock": False, "graphics": "", "music": ""})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    outs = sorted((folder / "Output").glob("*.mp4"))
    assert outs
    probe = ffmpeg.probe(outs[-1])
    v = ffmpeg.video_stream(probe)
    w, h = (int(x) for x in size.split("x"))
    assert (v["width"] > v["height"]) == (w > h)
    assert v.get("color_space") == "bt709" and any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in probe["streams"])
    r = subprocess.run([FF, "-hide_banner", "-i", str(outs[-1]), "-vf", "blackdetect=d=0.3:pix_th=0.05,freezedetect=n=-60dB:d=1.5", "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=120).stderr
    print(r[-900:])
    assert "black_start" not in r, "a black stretch in the delivered file"
    assert "freeze_start" not in r, "a frozen stretch in the delivered file"
    # an older plan: no sids, no style object, no file seconds
    st = LP2.State(LP2.home_of(projects.get(proj["id"])))
    plan = st.get("plan.json")
    plan.pop("style_obj", None)
    for s in plan.get("pool") or []:
        for k in ("sid", "file", "fstart", "fend"):
            s.pop(k, None)
    st.put("plan.json", plan)
    runner.rerender(proj["id"])
    assert store.get(proj["job"])["status"] == "done"
    out2 = sorted((folder / "Output").glob("*.mp4"))[-1]
    assert abs(ffmpeg.duration(out2) - ffmpeg.duration(outs[-1])) < 2.0
