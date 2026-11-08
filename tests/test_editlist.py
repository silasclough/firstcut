"""Exact edit lists: which file an entry names, an outline whose numbers are file names becoming an edit list by itself, the B-roll rules,
and a real render that plays exactly the clips given, in order, trimmed, with the silent ones silent."""
import json
import subprocess

import pytest

from clipfarm.clipper import editlist as EL
from clipfarm.clipper import flow as FL


def F(name, dur=10.0, start=0.0):
    return {"name": name, "rel": name, "dur": dur, "start": start, "end": start + dur, "audio": True, "w": 320, "h": 240, "fps": 30.0}


FILES = [F("1.01 Welcome.mov", start=0), F("1.02 Why we came.mov", start=20), F("2.01 Prayer seminar.mov", start=40), F("2.02 Optional song.mov", start=60),
         F("1.011 Other.mov", start=80), F("8.02 Worship.mov", start=100)]


def test_an_entry_names_exactly_one_file():
    assert EL.find_file("1.01", FILES)[0]["name"] == "1.01 Welcome.mov"                 # a leading number: "1.01" is not "1.011"
    assert EL.find_file("2.01 prayer seminar", FILES)[0]["name"] == "2.01 Prayer seminar.mov"
    assert EL.find_file("welcome", FILES)[0]["name"] == "1.01 Welcome.mov"
    f, cands = EL.find_file("1.0", FILES)                                               # two files could be meant: a question, not a guess
    assert f is None and len(cands) >= 2
    assert EL.find_file("9.99", FILES) == (None, [])


def test_parse_checks_the_list():
    assert EL.parse(["1.01", {"file": "1.02", "start": 2, "end": 5, "audio": "silent"}])[1]["audio"] == "silent"
    for bad in ([{"file": ""}], [{"file": "x", "start": 5, "end": 2}], [{"file": "x", "audio": "loud"}], "not a list", [3]):
        with pytest.raises(ValueError):
            EL.parse(bad)


def test_items_follow_the_list_and_say_what_is_wrong():
    items, problems, how = EL.items_for(EL.parse([{"file": "2.01", "start": 1, "end": 4}, "1.01", {"file": "8.02", "audio": "silent"}, {"file": "1.02", "end": 50}]), FILES, 5.0)
    assert [i["file"] for i in items] == ["2.01 Prayer seminar.mov", "1.01 Welcome.mov", "8.02 Worship.mov", "1.02 Why we came.mov"]    # the order given
    assert (items[0]["start"], items[0]["end"]) == (41.0, 44.0)                          # trims are inside the clip (its place on the timeline + seconds in)
    assert how["8.02 Worship.mov"] == "silent" and how["1.01 Welcome.mov"] == "own"
    assert any("only 3.0 s" in p for p in problems) and any("only 10.0 s" in p or "ends at 50" in p for p in problems)
    with pytest.raises(ValueError, match="can't be found"):
        EL.items_for(EL.parse(["1.01", "7.77"]), FILES)


def test_an_outline_numbered_like_the_files_is_an_edit_list_by_itself():
    notes = "Make a 4 minute video.\n1.01 Welcome from the director\n1.02 Why we came\n2.01 The prayer seminar\n2.02 Optional song\nNo captions."
    steps = FL.steps(notes)
    got = EL.from_steps(steps, FILES)
    assert [e["file"] for e in got] == ["1.01 Welcome.mov", "1.02 Why we came.mov", "2.01 Prayer seminar.mov", "2.02 Optional song.mov"]
    assert EL.from_steps(FL.steps("1. Welcome\n2. Why we came\n3. The prayer seminar"), FILES) is None       # numbers that are not file names: an outline about what is said
    assert EL.from_steps(FL.steps("1.01 a\n1.02 b\n9.01 c\n9.02 d\n9.03 e"), FILES) is None                 # most of it is not in the folder


def test_the_broll_that_may_be_used():
    pool = [{"file": "a.mov", "what": "students walking", "start": 0, "end": 4}, {"file": "b.mov", "what": "the Liberia prison yard", "start": 0, "end": 5},
            {"file": "c.mov", "what": "a quick pan", "start": 0, "end": 0.9}, {"file": "d.jpg", "what": "a photo", "photo": True, "start": 9e6, "end": 9e6 + 4}]
    got = EL.clean_pool(pool, ["prison yard"], 1.5)
    assert [p["file"] for p in got] == ["a.mov", "d.jpg"]
    assert len(EL.clean_pool(pool, None, 0)) == 4


# ------------------------------------------------------------------------------------------- a real render
def test_the_cut_plays_exactly_the_list(tmp_path, monkeypatch):
    from test_aivideo import _FakeAI, _Res

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core import ffmpeg
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    ff = ffmpeg.bin_path("ffmpeg")
    folder = tmp_path / "Raw"
    folder.mkdir()

    def clip(name, color, secs, sound):
        cmd = [ff, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:size=320x240:rate=30:duration={secs}"]
        if sound:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=400:duration={secs}", "-c:a", "aac", "-shortest"]
        subprocess.run([*cmd, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(folder / name)], check=True, timeout=120)
    clip("1.01 Welcome.mov", "red", 8, True)
    clip("1.02 Prayer.mov", "green", 8, True)
    clip("8.02 Worship.mov", "blue", 6, True)                              # has sound, but the list says silent
    clip("Spare.mov", "white", 6, False)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            self.calls.append(task)
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    fake = AI()
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Seminar", "", {"transcriber": "openai"}, {"clips": 0})
    edit = [{"file": "1.02", "start": 1, "end": 4}, {"file": "1.01", "end": 5}, {"file": "8.02", "audio": "silent"}]
    projects.prepare_local(proj["id"], {"edit_list": edit, "broll_mode": "none", "caption_style": "none", "music": "", "graphics": "", "library": False})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    out = next((folder / "Output").glob("*.mp4"))
    assert abs(ffmpeg.duration(out) - (3 + 5 + 6)) < 0.4                  # 3 s + 5 s + 6 s, in the order given, nothing else
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["exact"] and [e["file"] for e in plan["edit_list"]] == ["1.02 Prayer.mov", "1.01 Welcome.mov", "8.02 Worship.mov"]
    assert not any(t in fake.calls for t in ("story_edit", "speaker_roles", "video_length", "broll_plan"))        # nothing was planned, nothing listened to
    files = json.loads((folder / ".aivideo" / "files.json").read_text())
    assert next(f for f in files if f["name"] == "8.02 Worship.mov")["audio"] is False
    # the colours play in the list's order: green, then red, then blue (frames at 1 s, 5 s and 11 s)
    def pixel(t):
        raw = subprocess.run([ff, "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, timeout=60).stdout
        return tuple(raw[:3])
    g, r, b = pixel(1.0), pixel(5.0), pixel(11.0)
    assert g[1] > g[0] and r[0] > r[1] and b[2] > b[0]
    audio = subprocess.run([ff, "-hide_banner", "-nostats", "-ss", "9", "-t", "4", "-i", str(out), "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, timeout=60).stderr
    import re
    peak = re.findall(r"max_volume: (-?[\d.]+|-inf) dB", audio)
    assert peak and (peak[-1] == "-inf" or float(peak[-1]) < -60)                                           # the silent clip is silent
    early = subprocess.run([ff, "-hide_banner", "-nostats", "-ss", "0.5", "-t", "2", "-i", str(out), "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, timeout=60).stderr
    assert float(re.findall(r"max_volume: (-?[\d.]+) dB", early)[-1]) > -30                              # the clips with their own sound have it
