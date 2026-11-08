"""Notes for a finished long video (2026-11-06): YouTube chapters, a description, and a quality check with timecodes and what to do."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

from clipfarm.clipper import notes as NT
from clipfarm.clipper import qa
from clipfarm.core import ffmpeg

FF = ffmpeg.bin_path("ffmpeg")


def W(text, a, b):
    return {"text": text, "start": a, "end": b}


def talk(dur=120.0):
    """A sentence every 4 s for `dur` seconds, about different things."""
    out, t, k = [], 0.0, 0
    while t < dur - 4:
        for i, w in enumerate(f"Topic {k} sentence about thing number {k} here.".split()):
            out.append(W(w, t + i * 0.4, t + i * 0.4 + 0.35))
        t += 4.0
        k += 1
    return out


def test_youtube_rules_and_tidying():
    assert NT.youtube_ok([(0, "Intro"), (12, "The plan"), (40, "How it went")], 60)
    assert not NT.youtube_ok([(5, "Intro"), (20, "B"), (40, "C")], 60)                    # first not at 0:00
    assert not NT.youtube_ok([(0, "A"), (8, "B"), (40, "C")], 60)                         # under 10 s
    assert not NT.youtube_ok([(0, "A"), (30, "B")], 60)                                   # fewer than 3
    t = NT.tidy([(3, "Intro"), (8, "Too close"), (25, "Middle"), (55, "Too late"), (40, "  Late   part ")], 60)
    assert t == [(0.0, "Intro"), (25, "Middle"), (40, "Late part")] and NT.youtube_ok(t, 60)
    assert NT.chapters_txt(t) == "00:00 Intro\n00:25 Middle\n00:40 Late part\n" and NT.clock(3725) == "1:02:05"


def test_chapters_come_from_the_ai_and_fall_back_to_the_transcript():
    words = talk(180)

    class Good:
        def complete_json(self, task, system, prompt, schema):
            assert task == "video_chapters" and "Northfield" in prompt                           # the glossary goes along
            return SimpleNamespace(data={"summary": "Students at Northfield talk about faith.", "chapters": [
                {"at": 2, "title": "Why they came"}, {"at": 61, "title": "Classes"}, {"at": 120, "title": "What's next"}]})
    chs, summary = NT.chapters(Good(), words, 180.0, ["Northfield"])
    assert chs == [(0.0, "Why they came"), (61, "Classes"), (120, "What's next")] and summary.startswith("Students")

    class Bad:
        def complete_json(self, *a, **k):
            return SimpleNamespace(data={"summary": "x", "chapters": [{"at": 0, "title": "Only one"}]})
    chs, _ = NT.chapters(Bad(), words, 180.0)
    assert NT.youtube_ok(chs, 180.0) and len(chs) >= 3                                      # breaks the rules: from the transcript

    class Down:
        def complete_json(self, *a, **k):
            raise RuntimeError("offline")
    chs, summary = NT.chapters(Down(), words, 180.0)
    assert NT.youtube_ok(chs, 180.0) and summary == "" and chs[0][1].startswith("Topic 0")
    assert NT.chapters(None, talk(20), 20.0) == ([], "")                                    # too short for chapters


def test_the_render_s_own_findings_are_collected_per_render():
    NT.begin()
    NT.note("warn", "a shot under a second that couldn't be covered", 12.0, 13.0, do="Extend it.")
    got = NT.take()
    assert got == [{"level": "warn", "what": "a shot under a second that couldn't be covered", "start": 12.0, "end": 13.0, "do": "Extend it."}]
    assert NT.take() == []


def test_the_report_says_how_bad_where_and_what_to_do():
    issues = [{"level": "info", "what": "2 clips shot in S-Log3", "do": "Grade them in Final Cut."},
              {"level": "bad", "what": "the sound clips", "start": 10.0, "end": 12.1, "do": "Lower it."},
              {"level": "warn", "what": "the picture doesn't move for 6 s", "start": 7.0, "end": 13.0}]
    md = NT.quality_md(issues, {"duration": 20, "lufs": -14.0, "peak": -1.5, "cuts": 3, "longest_hold": 9}, "Talk v2")
    assert NT.counts(issues) == {"fix": 1, "look": 1, "info": 1} and NT.headline(NT.counts(issues)) == "1 to fix, 1 to look at"
    lines = [x for x in md.splitlines() if x.startswith("- ")]
    assert lines[0].startswith("- **Fix** · 00:10–00:12 · The sound clips") and "What to do: Lower it." in md
    assert lines[1].startswith("- **Look** · 00:07–00:13") and lines[2].startswith("- **Info** · whole video")
    assert NT.headline(NT.counts([])) == "Nothing to fix"


def test_a_finished_video_with_a_freeze_and_clipping_is_flagged_where_it_happens(tmp_path):
    v = tmp_path / "v.mp4"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=20", "-f", "lavfi", "-i", "sine=f=440:d=20",
                    "-filter_complex", "[0:v]trim=0:6,setpts=PTS-STARTPTS[a];[0:v]trim=6:7,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=6[b];"
                    "[0:v]trim=13:20,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0[v];[1:a]volume='if(between(t,10,12),8,0.2)':eval=frame[au]",
                    "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(v)], check=True)
    r = qa.check(v)
    clip = [i for i in r["issues"] if "clips" in i["what"]]
    froze = [i for i in r["issues"] if "doesn't move" in i["what"]]
    assert clip and clip[0]["level"] == "bad" and 9.5 <= clip[0]["start"] <= 10.5 and 11.5 <= clip[0]["end"] <= 12.6
    assert froze and 6.5 <= froze[0]["start"] <= 7.5 and 12.5 <= froze[0]["end"] <= 13.5
    assert r["peak"] > -1 and all(i.get("do") for i in r["issues"])


def test_variable_frame_rate_and_mixed_sources_are_noted(tmp_path):
    vfr = tmp_path / "phone.mp4"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=6",
                    "-vf", "select='if(lt(t,3),1,not(mod(n,3)))'", "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vfr)], check=True)
    cfr = tmp_path / "camera.mp4"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cfr)], check=True)
    issues = NT.sources([{"path": str(vfr)}, {"path": str(cfr)}, {"path": str(tmp_path / "pic.jpg"), "photo": True}])
    whats = " | ".join(i["what"] for i in issues)
    assert "phone.mp4 has a variable frame rate" in whats and "mixes frame rates" in whats and "mixes picture sizes" in whats
    assert "camera.mp4 has a variable" not in whats


def test_a_plain_lecture_with_no_broll_is_not_a_bad_quality_issue(tmp_path):
    import subprocess

    from clipfarm.clipper import qa
    from clipfarm.core import ffmpeg
    v = tmp_path / "lecture.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:size=160x120:rate=10:duration=50",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=50", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(v)], check=True)
    for profile in (None, "seminar", "interview_clip", "longform_talk"):
        rep = qa.check(v, None, profile=profile)
        assert not [i for i in rep["issues"] if i["level"] == "bad" and "never changes" in i["what"]], profile
    calm = qa.check(v, None, profile="seminar")
    assert not [i for i in calm["issues"] if "never changes" in i["what"] or "the same picture for" in i["what"]]
