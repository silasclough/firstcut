import json
import subprocess
import zipfile
from fractions import Fraction

import pytest

from clipfarm.clipper import fcpxml as FX
from clipfarm.clipper import polish as P
from clipfarm.clipper.engine import Engine
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config


def test_keep_intervals_talking_vs_visual():
    talk = [{"text": "w", "start": t, "end": t + 0.3} for t in (1.0, 1.5, 2.0, 5.0, 5.4)]   # 2.7 s silence
    iv = P.keep_intervals(talk, 8.0, 0.75)
    assert len(iv) == 2 and iv[0][1] < 3.0 < 4.5 < iv[1][0]
    sparse = [{"text": "wow", "start": 50.0, "end": 50.4}]                                 # mostly visual
    assert P.keep_intervals(sparse, 120.0, 0.75) == [(49.0, 120.0)]
    assert P.select_expr([(1, 2), (3.5, 4)]) == "between(t,1.000,2.000)+between(t,3.500,4.000)"


def test_fcpxml_is_valid_for_final_cut(tmp_path):
    src = FX.Media("/Volumes/X/My Project/Media/source.mp4", 120.0, fps=Fraction(30000, 1001))
    bed = FX.MusicBed("/Volumes/X/My Project/Music/a b.mp3", 200.0, "a b", start=10,
                      keyframes=[(0, -14), (1, -24), (3, -24), (3.3, -14)])
    xml = FX.build("My Project", src, [(0.5, 10.0), (11.2, 30.0)], 1920, 1080, Fraction(30000, 1001),
                   captions=[(0.1, 1.0, "hello & bye"), (9.0, 10.5, "over the cut")], music=bed, hook=("Hook", 3))
    p = tmp_path / "p.fcpxml"
    p.write_text(xml)
    assert "file:///Volumes/X/My%20Project/Media/source.mp4" in xml and 'lane="-1"' in xml
    assert xml.count("<asset-clip ref=\"r3\"") == 2 and xml.count('name="Caption"') == 2
    if not FX.DTD_DIR.exists():
        pytest.skip("Final Cut Pro not installed here - DTD check skipped")
    assert FX.validate(p) == []


def _media(path, seconds, audio=True, w=320, h=240):
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=30"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    cmd += (["-c:a", "aac", "-shortest"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def test_polish_end_to_end_with_music_and_fcp(tmp_path):
    cfg = load_config()
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=500:duration=30",
                    "-c:a", "libmp3lame", str(music_dir / "lofi - Test Track - Someone.mp3")], check=True)
    (music_dir / "lofi - Test Track - Someone.json").write_text(json.dumps(
        {"title": "Test Track", "artist": "Someone", "license": "CC0 1.0", "credit": "Music: Test Track (CC0)",
         "moods": ["chill"], "genre": "lofi"}))
    cfg.data["paths"].update({"music": str(music_dir), "state": str(tmp_path / "state")})
    cfg.data["clipper"]["fcp_unpack_dir"] = str(tmp_path / "Unpacked")
    store = Store(tmp_path / "c")
    eng = Engine(cfg, store)
    job = store.create({"filename": "talk.mp4"}, {"mode": "polish", "music": "chill", "fcp": True,
                                                  "caption_style": "bold", "pace": "natural"})
    src = _media(store.dir(job["id"]) / "source.mp4", 6)
    words = [{"text": t, "start": a, "end": a + 0.35} for t, a in
             [("hello", 0.5), ("there", 0.9), ("friend", 1.3), ("after", 3.6), ("a", 4.0), ("pause", 4.3)]]
    clip = P.run(eng, job["id"], src, words)
    d = store.dir(job["id"])
    assert clip["polish"] and clip["cuts"] == 1 and clip["trimmed_s"] > 1.0
    assert ffmpeg.duration(d / "clip_01.mp4") == pytest.approx(clip["duration"], abs=0.25)
    assert clip["music"]["title"] == "Test Track" and "Test Track" in (d / "clip_01.txt").read_text()
    names = zipfile.ZipFile(d / "fcp_project.zip").namelist()
    folder = clip["fcp"]["folder"]
    assert f"{folder}/{folder}.fcpxml" in names and f"{folder}/Media/source.mp4" in names
    assert any(n.startswith(f"{folder}/Music/") and n.endswith(".mp3") for n in names)
    assert any(n.endswith("LICENSE.txt") for n in names)
    xml = (d / "fcp" / folder / f"{folder}.fcpxml").read_text()
    assert str(tmp_path / "Unpacked").replace(" ", "%20") in xml and xml.count('ref="r3"') == 2
    if FX.DTD_DIR.exists():
        assert clip["fcp"]["valid"], clip["fcp"]["errors"]


# ----------------------------------------------------------------------------- long-video mode (several clips -> one story)
class _Res:
    def __init__(self, data):
        self.data = data


class _FakeEditor:
    """Keeps the first and last sentence of every chunk it is shown (drops the rambling middle)."""
    def __init__(self):
        self.prompts = []

    def complete_json(self, task, system, prompt, schema):
        import re
        self.prompts.append(prompt)
        idx = [int(m) for m in re.findall(r"^\s+\[(\d+)\]", prompt, re.M)]
        return _Res({"title": "A day out", "keep": [{"first": idx[0], "last": idx[0]}, {"first": idx[-1], "last": idx[-1]}],
                     "summary": f"kept {idx[0]} and {idx[-1]}"})


class _Engine:
    def __init__(self, llm):
        self.llm = llm

    def router_for_now(self):
        return self.llm, "fake"

    def creative_router(self):
        return self.llm


def _speech(start, n, step=0.5):
    return [{"text": f"w{int(start * 10) + i}" + ("." if i % 6 == 5 else ""), "start": start + i * step, "end": start + i * step + 0.3}
            for i in range(n)]


def test_long_mode_keeps_story_order_drops_the_middle_and_keeps_visual_clips_whole():
    from clipfarm.clipper import story
    files = [{"name": "a.mov", "start": 0.0, "end": 40.0}, {"name": "b.mov", "start": 40.0, "end": 100.0},
             {"name": "c.mov", "start": 100.0, "end": 160.0}]
    words = _speech(1.0, 70) + _speech(101.0, 80)                    # a, c talk; b is just pictures (no speech)
    llm = _FakeEditor()
    plan = story.plan(_Engine(llm), words, files, 160.0, "IMG title", target_s=100.0, gap=0.6)
    ivs = plan.intervals
    assert ivs == sorted(ivs) and all(a < b for a, b in ivs)                    # strictly in recording order
    assert any(a <= 40.0 and b >= 100.0 or (a >= 39.9 and b >= 99.9) for a, b in ivs) or sum(b - a for a, b in ivs if 40 <= a < 100) >= 55
    talk_kept = sum(b - a for a, b in ivs if a < 40 or a >= 100)
    assert talk_kept < 0.7 * (35 + 40)                                           # the rambling middle of the talking clips is gone
    assert plan.title == "A day out" and plan.summary
    by = {f["name"]: f["kept_s"] for f in plan.kept_files}
    assert by["b.mov"] >= 55                                                     # the picture-only clip is kept whole
    assert "Story so far" in llm.prompts[-1] or len(llm.prompts) == 1


def test_long_mode_without_a_working_ai_keeps_everything_in_order():
    from clipfarm.clipper import story

    class Broken:
        def complete_json(self, *a, **k):
            raise RuntimeError("no model")
    files = [{"name": "a.mov", "start": 0.0, "end": 60.0}]
    plan = story.plan(_Engine(Broken()), _speech(1.0, 100), files, 60.0, "t", target_s=30.0, gap=0.6)
    assert sum(b - a for a, b in plan.intervals) > 40                            # nothing silently thrown away


def test_long_mode_never_drops_a_whole_clip():
    from clipfarm.clipper import story

    class KeepOnlyFirst:
        def complete_json(self, task, system, prompt, schema):
            import re
            idx = [int(m) for m in re.findall(r"^\s+\[(\d+)\]", prompt, re.M)]
            return _Res({"title": "t", "keep": [{"first": idx[0], "last": idx[3]}], "summary": "s"})   # ignores later clips
    files = [{"name": "a.mov", "start": 0.0, "end": 60.0}, {"name": "b.mov", "start": 60.0, "end": 120.0}]
    plan = story.plan(_Engine(KeepOnlyFirst()), _speech(1.0, 100) + _speech(61.0, 100), files, 120.0, "t", target_s=60.0, gap=0.6)
    by = {f["name"]: f["kept_s"] for f in plan.kept_files}
    assert by["b.mov"] >= 12                                                   # clip b still has its share


def test_the_users_brief_reaches_the_ai():
    from clipfarm.clipper import story
    llm = _FakeEditor()
    files = [{"name": "a.mov", "start": 0.0, "end": 60.0}]
    story.plan(_Engine(llm), _speech(1.0, 100), files, 60.0, "t", target_s=30.0, gap=0.6, brief="only the funny reactions")
    assert "only the funny reactions" in llm.prompts[0]


def test_whole_video_edit_can_move_a_part_to_the_front(tmp_path):
    import subprocess
    from clipfarm.clipper import polish
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core import ffmpeg
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {"mode": "polish", "caption_style": "none", "music": "", "fcp": False, "pace": "natural"})
    src = store.dir(job["id"]) / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=30",
                    "-f", "lavfi", "-i", "sine=f=300:d=30", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    words = [{"text": f"w{i}", "start": 0.2 + i * 0.5, "end": 0.6 + i * 0.5} for i in range(58)]
    clip = polish.run(Engine(cfg, store), job["id"], src, words, override=[(20.0, 30.0), (5.0, 15.0)])    # the end first, then a middle part
    assert 19 < clip["duration"] < 21 and clip["cuts"] == 1
    assert 19 < ffmpeg.duration(store.dir(job["id"]) / clip["file"]) < 21
