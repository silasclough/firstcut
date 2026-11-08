"""WP3 (docs/plans/editor-improvement-plan.md): which footage may be cut to as B-roll, which files may be camera angles, what the person ruled out
(words, files, stretches), photos found anywhere, loose music, wrong-shaped shots and library shots that remember what they were found for.
The pure rules (poolrules.py) are tested without a render; the rest run small real folders with a faked AI (offline)."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from clipfarm.clipper import aivideo as AV
from clipfarm.clipper import broll as BR
from clipfarm.clipper import editlist as EL
from clipfarm.clipper import localproj as LP
from clipfarm.clipper import multicam as MC
from clipfarm.clipper import photos as PH
from clipfarm.clipper import poolrules as PR
from clipfarm.clipper.engine import Engine
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config
from test_aivideo import _FakeAI, _Res, _keep_cut_folder

FF = ffmpeg.bin_path("ffmpeg")


# --------------------------------------------------------------------------------------------------------------------------- helpers
def _f(name, role="", start=0.0, dur=20.0, broll=False, audio=True, w=1920, h=1080, **facts):
    rec = {"name": name, "path": f"/x/{name}", "dur": dur, "start": start, "end": start + dur, "w": w, "h": h, "fps": 25.0, "audio": audio, "broll": broll,
           "size": 1000}
    if role or facts:
        rec["facts"] = {"role": role, **facts}
    return rec


def _sc(file, start, end, what="a shot", quality=7, **kw):
    return {"start": start, "end": end, "what": what, "quality": quality, "file": file, **kw}


def _sparse_always(f, words):
    return True


def _clip(path: Path, secs: int, audio: bool, size="320x240", color="blue") -> None:
    cmd = [FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:size={size}:rate=30:duration={secs}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True, timeout=120)


def _talk(path: Path, secs: int) -> None:
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30:duration={secs}", "-f", "lavfi", "-i",
                    f"sine=frequency=300:duration={secs}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
                   check=True, timeout=120)


def _runner(tmp_path, monkeypatch, ai):
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    monkeypatch.setattr(LP, "LocalAI", lambda c: ai)
    return cfg, store, projects, LP.LocalRunner(cfg, store, projects)


def _file_in(prompt: str) -> str:
    import re
    m = re.search(r"File: (.+?) \(", prompt)
    return m.group(1) if m else ""


TALK_WORDS = [{"text": f"w{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(22)]


# ------------------------------------------------------------------------------------------------------------------ pool membership
def test_a_talking_head_scene_is_never_in_the_broll_pool():
    files = [_f("talk.mov", "a_roll", 0, 600), _f("okay.mov", "a_roll", 600, 5), _f("wall.mov", "broll", 605, 12, audio=False)]
    scenes = {"okay.mov": [_sc("okay.mov", 600, 605, "the speaker says okay")],               # the audit's case: a short clip of the speaker with a few words
              "wall.mov": [_sc("wall.mov", 605, 610, "a man talking to the camera", is_talking_head=True, usable=True),
                           _sc("wall.mov", 610, 617, "a red wall", usable=True)]}
    got = PR.select(files, scenes, [], _sparse_always)
    assert 1 not in got["named"] | got["silent"] and "okay.mov" not in got["scenes"]
    assert [s["what"] for s in got["scenes"]["wall.mov"]] == ["a red wall"]
    assert got["silent"] == {2}
    assert "speaker" in " ".join(e["why"] for e in got["excluded"])
    assert "speaker" in PR.excluded_line(got["excluded"]) and PR.excluded_line([]) == ""


def test_a_clip_of_the_speaker_with_few_words_is_never_offered_as_broll(tmp_path, monkeypatch):
    """End to end: talk.mov, a 4 s clip of the speaker saying one word, a B-roll file. The speaker's clip is not in the pool."""
    folder = tmp_path / "Lecture"
    (folder / "B-roll").mkdir(parents=True)
    _talk(folder / "talk.mov", 12)
    _talk(folder / "okay.mov", 4)
    _clip(folder / "B-roll" / "red wall.mov", 6, audio=False, color="red")
    seen = []

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_describe":                                                  # (the words alone make a one-word clip look like B-roll: the picture model says it is the speaker)
                seen.append(_file_in(prompt))
                if _file_in(prompt) == "okay.mov":
                    return _Res({"scenes": [{"start_s": 0, "end_s": 4, "what": "the speaker says okay to the camera", "quality": 8, "is_talking_head": True}]})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    cfg, store, projects, runner = _runner(tmp_path, monkeypatch, AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: TALK_WORDS if f["name"] == "talk.mov" else [{"text": "okay.", "start": 1.0, "end": 1.3}])
    proj = projects.create_local(str(folder), "Lecture", "", {"transcriber": "openai"}, {"clips": 0})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done", store.get(proj["job"])
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert {s["file"] for s in plan["pool"]} == {"red wall.mov"}
    assert any(e["file"] == "okay.mov" and "speaker" in e["why"] for e in plan.get("pool_excluded", []))      # (for the report: what was kept out of the B-roll)


def test_a_big_face_in_a_file_with_speech_is_a_talking_head_even_when_the_model_did_not_say_so():
    f = _f("okay.mov", "broll", 600, 5, facts_speech := None) if False else _f("okay.mov", "broll", 600, 5, speech={"ratio": 0.08})
    assert PR.talking_head(f, _sc("okay.mov", 600, 605, face_ratio=0.45))
    assert not PR.talking_head(f, _sc("okay.mov", 600, 605, face_ratio=0.1))                      # a person far away
    assert not PR.talking_head(_f("quiet.mov", "broll", speech={"ratio": 0.0}), _sc("quiet.mov", 0, 4, face_ratio=0.45))   # a silent portrait shot is B-roll
    got = PR.select([_f("talk.mov", "a_roll", 0, 600), f], {"okay.mov": [_sc("okay.mov", 600, 605, face_ratio=0.45)]}, [], _sparse_always)
    assert "okay.mov" not in got["scenes"] and got["excluded"]


def test_a_named_broll_file_that_was_not_described_stays_broll_with_a_placeholder_scene():
    f = _f("mystery.mov", "broll", 100, 12, broll=True, audio=False, usable=[{"start": 0.0, "end": 3.0}, {"start": 5.0, "end": 9.0}])
    got = PR.select([_f("talk.mov", "a_roll", 0, 100), f], {}, [], _sparse_always)
    assert got["named"] == {1} and got["silent"] == set()                                # B-roll, and never a shot that plays on its own
    sc = got["scenes"]["mystery.mov"]
    assert [(s["fstart"], s["fend"], s["start"], s["end"]) for s in sc] == [(0.0, 3.0, 100.0, 103.0), (5.0, 9.0, 105.0, 109.0)]
    assert all(s["what"].startswith("(not described)") and s["quality"] == 4.0 and s["described"] is False for s in sc)
    # without a footage record the whole file is the placeholder
    g2 = PR.select([_f("blank.mov", broll=True, dur=8, audio=False)], {}, [], _sparse_always)
    assert [(s["fstart"], s["fend"]) for s in g2["scenes"]["blank.mov"]] == [(0.0, 8.0)]


def test_a_placeholder_scene_is_in_the_plans_pool_when_the_vision_model_failed(tmp_path, monkeypatch):
    folder = tmp_path / "Shop"
    (folder / "B-roll").mkdir(parents=True)
    _talk(folder / "talk.mov", 12)
    _clip(folder / "B-roll" / "mystery.mov", 6, audio=False)

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_describe":
                raise RuntimeError("the picture model is down")
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    cfg, store, projects, runner = _runner(tmp_path, monkeypatch, AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: TALK_WORDS)
    proj = projects.create_local(str(folder), "Shop", "", {"transcriber": "openai"}, {"clips": 0})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done", store.get(proj["job"])
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    ph = [s for s in plan["pool"] if s["file"] == "mystery.mov"]
    assert ph and all(s["what"].startswith("(not described)") and s["quality"] == 4.0 for s in ph)
    assert [it["kind"] for it in plan["items"]].count("visual") == 0                      # (never played on its own as a story shot)


# ---------------------------------------------------------------------------------------------------------------------- multicam
def test_a_broll_clip_that_heard_the_room_is_not_a_camera_angle():
    a = _f("camA.mov", "a_roll", dur=300, faces={"on_camera_speaker_ratio": 0.9})
    b = _f("camB.mov", "a_roll", dur=300, faces={"on_camera_speaker_ratio": 0.8})
    wide = _f("wide.mov", "broll", dur=30)                                                # a B-roll clip whose microphone heard the room
    cand, note = MC.angle_candidates([a, b, wide])
    assert [f["name"] for f in cand] == ["camA.mov", "camB.mov"] and note == ""
    # only one file shows the speaker: the sound test decides, B-roll still cannot be an angle, and the job says so
    wide2 = _f("wide2.mov", "voiceover_broll", dur=300, faces={"on_camera_speaker_ratio": 0.05})
    cand2, note2 = MC.angle_candidates([a, wide2, wide])
    assert [f["name"] for f in cand2] == ["camA.mov", "wide2.mov"] and "sound" in note2
    # no footage record at all: the sound test of before
    cand3, note3 = MC.angle_candidates([_f("x.mov"), _f("y.mov"), _f("z.mov", broll=True)])
    assert [f["name"] for f in cand3] == ["x.mov", "y.mov"] and note3 == ""


def test_cameras_of_different_families_never_share_an_angle_and_the_director_is_told_what_each_shows():
    g = _f("C0001.MP4", "a_roll", camera={"family": "gopro"})
    s = _f("C0001 .MP4", "a_roll", camera={"family": "sony_c"})
    assert MC.signature(g) != MC.signature(s)
    clips = [{"name": "tight.mov"}, {"name": "tight2.mov"}]
    by = {"tight.mov": _f("tight.mov", faces={"largest_ratio_median": 0.45, "count_median": 1, "measured": True}),
          "tight2.mov": _f("tight2.mov", faces={"largest_ratio_median": 0.5, "count_median": 1, "measured": True})}
    assert "one person" in MC.angle_shows(clips, by)
    assert "wide" in MC.angle_shows([{"name": "room.mov"}], {"room.mov": _f("room.mov", faces={"largest_ratio_median": 0.0, "count_median": 0, "measured": True})})
    assert MC.angle_shows([{"name": "n.mov"}], {"n.mov": _f("n.mov")}) == ""


# -------------------------------------------------------------------------------------------------------- what the person ruled out
def test_avoid_words_and_excluded_files_apply_in_story_keep_cut_and_render(tmp_path, monkeypatch):
    # (1) the story plan
    folder = tmp_path / "Prison visit"
    (folder / "B-roll").mkdir(parents=True)
    _talk(folder / "talk.mov", 12)
    for n in ("prison yard", "f03", "red wall"):
        _clip(folder / "B-roll" / f"{n}.mov", 6, audio=False)

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_describe":
                return _Res({"scenes": [{"start_s": 0, "end_s": 6, "what": "a place " + _file_in(prompt), "quality": 9}]})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    cfg, store, projects, runner = _runner(tmp_path, monkeypatch, AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: TALK_WORDS)
    proj = projects.create_local(str(folder), "Prison visit", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"avoid": ["prison yard"], "exclude_files": ["f03"]})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done", store.get(proj["job"])
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert {s["file"] for s in plan["pool"]} == {"red wall.mov"}
    assert plan["rules"]["avoid"] == ["prison yard"] and plan["rules"]["exclude_files"] == ["f03"]
    # (2) "add B-roll to a video"
    kdir = tmp_path / "k"
    kdir.mkdir()
    folder2 = _keep_cut_folder(kdir)
    cfg2, store2, projects2, runner2 = _runner(kdir, monkeypatch, AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: TALK_WORDS)
    proj2 = projects2.create_local(str(folder2), "Open house", "", {"transcriber": "openai"}, {"clips": 0})
    projects2.prepare_local(proj2["id"], {"keep_cut": True, "aroll": "Interview.mov", "graphics": "", "music": "", "library": False, "exclude_files": ["Campus"]})
    runner2.make(proj2["id"], {})
    assert store2.get(proj2["job"])["status"] == "done", store2.get(proj2["job"])
    plan2 = json.loads((folder2 / ".aivideo" / "plan.json").read_text())
    assert plan2["pool"] and {s["file"] for s in plan2["pool"]} == {"Walk and talk.mov"}
    # (3) render: library and stock scenes are added after the plan chose its pool, so the render clears them again with the same rules
    pool = [_sc("Library - Travel/prison yard.mov", 2e6, 2e6 + 5, "a prison yard at dawn", library=True),
            _sc("Library - Travel/f03.mov", 2e6 + 10, 2e6 + 15, "a field", library=True),
            _sc("Library - Travel/coast.mov", 2e6 + 20, 2e6 + 25, "the coast", library=True)]
    libfiles = [dict(_f(n, start=2e6 + 10 * k, dur=9, broll=True), library=True) for k, n in enumerate(
        ["Library - Travel/prison yard.mov", "Library - Travel/f03.mov", "Library - Travel/coast.mov"])]
    got, _ = LP.usable_pool(pool, libfiles, rules={"avoid": ["prison yard"], "exclude_files": ["f03.mov"]})
    assert [s["what"] for s in got] == ["the coast"]
    assert [s["what"] for s in EL.clean_pool(pool, ["prison yard"], 0.0, ["f03"])] == ["the coast"]


def test_an_exclude_window_keeps_broll_out_of_the_prayer():
    cut = [{"at": 4.0, "length": 2.0, "scene": 0}, {"at": 20.0, "length": 3.0, "scene": 1}, {"at": 31.0, "length": 2.5, "scene": 2}]
    kept, n = PR.outside_windows(cut, [[18.0, 30.0]])                                   # the prayer: 18 s to 30 s of the finished video
    assert [c["at"] for c in kept] == [4.0, 31.0] and n == 1
    kept2, n2 = PR.outside_windows(cut, [[5.5, 21.0]])                                  # a shot that only overlaps the window's edge goes too
    assert [c["at"] for c in kept2] == [31.0] and n2 == 2
    assert PR.outside_windows(cut, None) == (cut, 0)
    rules = PR.rules_from({"exclude_windows": [[1, 2], ["x", 3], [9, 4], [5]], "exclude_files": ["f1", ""], "avoid": "no list", "min_shot_s": "2"})
    assert rules == {"avoid": [], "exclude_files": ["f1"], "exclude_windows": [[1.0, 2.0]], "min_shot_s": 2.0}


def test_an_exclude_window_is_honoured_by_a_real_render(tmp_path, monkeypatch):
    folder = tmp_path / "Wedding"
    (folder / "B-roll").mkdir(parents=True)
    _talk(folder / "talk.mov", 14)
    _clip(folder / "B-roll" / "red wall.mov", 6, audio=False, color="red")
    cfg, store, projects, runner = _runner(tmp_path, monkeypatch, _FakeAI())               # (the fake plan puts one cutaway at 4.0 s)
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: TALK_WORDS)
    proj = projects.create_local(str(folder), "Wedding", "", {"transcriber": "openai"}, {"clips": 0})
    runner.make(proj["id"], {})
    base = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    assert base["cutaways"], "the fake planner should have placed a cutaway"
    p = json.loads((folder / ".aivideo" / "plan.json").read_text())
    p["rules"] = {**(p.get("rules") or {}), "exclude_windows": [[0.0, 60.0]]}                    # (the whole video: no B-roll anywhere)
    (folder / ".aivideo" / "plan.json").write_text(json.dumps(p))
    runner.rerender(proj["id"])
    after = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    assert not after["cutaways"]                                                          # (the speaker stays on screen in the whole window)


# ------------------------------------------------------------------------------------------------------------------------ photos
def test_photos_outside_a_photo_folder_are_found_and_only_fitting_ones_are_forced_in(tmp_path):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(1)
    folder = tmp_path / "Event"
    (folder / "Ceremony").mkdir(parents=True)
    (folder / "B-roll" / "photos").mkdir(parents=True)
    yy, xx = np.mgrid[0:800, 0:1200]
    photo = np.stack([xx * 255 // 1200, yy * 255 // 800, (xx + yy) * 255 // 2000], axis=-1).astype(np.uint8)
    photo = np.clip(photo.astype(int) + rng.integers(-20, 20, photo.shape), 0, 255).astype(np.uint8)                           # (many colours, like a photograph)
    Image.fromarray(photo).save(folder / "Ceremony" / "cake.jpg")                                                          # a real photo, in no photo folder
    Image.fromarray(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)).save(folder / "Ceremony" / "avatar.jpg")         # too small: not a photo for us
    Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(folder / "B-roll" / "photos" / "small.jpg")   # in a photo folder: counts whatever its size
    logo = Image.new("RGBA", (1200, 800), (0, 0, 0, 0))
    logo.paste((200, 0, 0, 255), (100, 100, 400, 400))
    logo.save(folder / "Ceremony" / "logo.png")                                                                            # a graphic: never a photo
    got = {p["name"] for p in PH.scan(folder, LP.skipped)}
    assert got == {"cake.jpg", "small.jpg"}
    good = {"what": "(still photo) a cake", "quality": 6.0, "raw_quality": 8.0}
    dull = {"what": "(still photo) a lamp post", "quality": 6.0, "raw_quality": 3.0}
    about = {"what": "(still photo) the old church", "quality": 6.0, "raw_quality": 3.0}
    said = PR.spoken_tokens("We married in the old church that summer.")
    assert PR.photo_fits(good, said) and not PR.photo_fits(dull, said) and PR.photo_fits(about, said)
    scenes = PH.scenes([{"name": "a.jpg", "path": str(folder / "Ceremony" / "cake.jpg"), "what": "a cake", "quality": 3.0}])
    assert scenes[0]["raw_quality"] == 3.0 and scenes[0]["quality"] == 6.0 and scenes[0]["w"] == 1200 and scenes[0]["h"] == 800


def test_photo_clips_are_baked_at_the_timeline_rate(tmp_path):
    pic = tmp_path / "p.jpg"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=800x600:rate=1", "-frames:v", "1", str(pic)], check=True)
    rate = LP.timeline_rate(23.976)
    assert rate == Fraction(24000, 1001)
    out = PH.bake(str(pic), "zoom_in", 1.5, 640, 360, float(rate), tmp_path / "p.mp4", motion_blur=False)
    info = subprocess.run([ffmpeg.bin_path("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout.strip()
    assert info == "24000/1001"


# ------------------------------------------------------------------------------------------------------------------------- music
def test_a_loose_music_file_is_a_song_not_a_recorder(tmp_path):
    folder = tmp_path / "Show"
    (folder / "Music").mkdir(parents=True)
    (folder / "Songs").mkdir()
    for p in (folder / "Music" / "own.wav", folder / "Songs" / "morning.wav", folder / "ZOOM0001.wav", folder / "Track 1.wav"):
        subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=8", str(p)], check=True)
    loose: list[Path] = []
    rec = LP.scan_audio(folder, music_out=loose)
    assert {a["name"] for a in rec} == {"ZOOM0001.wav", "Track 1.wav"}                  # (a recorder's channels are called Track 1: they stay recordings)
    assert [p.name for p in loose] == ["morning.wav"]
    assert sorted(t.path.name for t in LP.own_songs(folder)) == ["morning.wav", "own.wav"]


# ---------------------------------------------------------------------------------------------------------------------- orientation
def test_portrait_broll_is_marked_and_penalised_on_a_landscape_timeline():
    port = _sc("p.mov", 0, 4, "a man walks", quality=8, w=1080, h=1920)
    land = _sc("l.mov", 0, 4, "a street", quality=7, w=1920, h=1080)
    sq = _sc("s.mov", 0, 4, "a plate", quality=7, w=1080, h=1080)
    marked = PR.mark_orientation([port, land, sq], True)
    assert [bool(s.get("mismatch")) for s in marked] == [True, False, False]
    assert PR.effective_quality(marked[0]) == 6.0 and PR.effective_quality(marked[1]) == 7.0
    assert "[portrait]" in BR.scene_line(0, marked[0]) and "[portrait]" not in BR.scene_line(1, marked[1])
    flipped = PR.mark_orientation([port, land], False)                                                                  # a portrait timeline: the landscape shot is the wrong one
    assert [bool(s.get("mismatch")) for s in flipped] == [False, True]
    # the render's pool is marked the same way (the shot strip numbers its shots through the same function)
    files = [_f("talk.mov", "a_roll", dur=30, w=1920, h=1080)]
    pool, _ = LP.usable_pool([_sc("p.mov", 30, 34, "a man walks", quality=8, w=1080, h=1920, sid="s0000", fstart=0.0, fend=4.0)], files + [_f("p.mov", dur=4, start=30, w=1080, h=1920)])
    assert pool and pool[0].get("mismatch") is True


def test_the_planner_is_told_which_scenes_are_the_wrong_shape():
    seen = {}

    class LLM:
        def complete_json(self, task, system, prompt, schema, **kw):
            seen["prompt"] = prompt
            return _Res({"cutaways": []})
    from clipfarm.clipper import moments as M
    sents = [M.Sentence(0, 1.0, 5.0, "We walked through the market.")] if hasattr(M, "Sentence") else []
    pool = PR.mark_orientation([_sc("p.mov", 0, 4, "a man walks", w=1080, h=1920)], True)
    BR._plan_once(LLM(), sents, pool, talk_s=60.0)
    if sents:
        assert "[portrait]" in seen["prompt"] and "other way round" in seen["prompt"]


# ----------------------------------------------------------------------------------------------------------------------- library
def test_library_shots_keep_the_passage_they_were_found_for(tmp_path):
    class Cat:
        def enabled(self):
            return True

        def shots_for(self, ai, sents, brief, limit=0, progress=None):
            return [{"kind": "video", "id": "v1", "rel": "Trips/coast.mov", "path": "/x/coast.mov", "dur": 10.0, "start": 1.0, "end": 5.0, "what": "waves on rocks",
                     "quality": 8.0, "query": "the sea at dawn", "size": 5, "mtime": 5, "w": 1920, "h": 1080, "fps": 25.0},
                    {"kind": "video", "id": "v2", "rel": "Trips/dull.mov", "path": "/x/dull.mov", "dur": 10.0, "start": 2.0, "end": 6.0,
                     "what": "(found searching for 'the sea at dawn') Trips", "quality": 4.0, "described": False, "query": "the sea at dawn", "size": 6, "mtime": 6}]

        def thumb(self, *a, **k):
            raise KeyError

    cfg, store, projects, runner = _runner(tmp_path, pytest.MonkeyPatch(), _FakeAI())
    runner.catalog = Cat()
    st = LP.State(tmp_path / "state1")
    files, scenes = runner.add_library([], [], {}, _FakeAI(), st, "", lambda *a: None)
    sc = [s for v in scenes.values() for s in v]
    by_what = {s["what"]: s for s in sc}
    assert by_what["waves on rocks"]["found_for"] == "the sea at dawn" and "described" not in by_what["waves on rocks"]
    dull = next(s for s in sc if s["what"].startswith("(found searching"))
    assert dull["described"] is False and dull["found_for"] == "the sea at dawn"
    assert 'found for: "the sea at dawn"' in BR.scene_line(0, by_what["waves on rocks"])
    assert PR.effective_quality(dull) == dull["quality"] - 2.0
