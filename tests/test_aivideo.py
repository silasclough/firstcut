import json

import pytest

from clipfarm.clipper import aivideo as AV
from clipfarm.clipper import parakeet as PK
from clipfarm.clipper.store import Store
from clipfarm.core.config import load_config


def test_parakeet_tokens_become_words():
    data = {"sentences": [{"tokens": [
        {"text": " Hel", "start": 0.0, "end": 0.2}, {"text": "lo", "start": 0.2, "end": 0.4},
        {"text": " there", "start": 0.5, "end": 0.8}, {"text": ".", "start": 0.8, "end": 0.85}, {"text": " ", "start": 0.9, "end": 0.9}]}]}
    w = PK.words_from_json(data)
    assert [x["text"] for x in w] == ["Hello", "there."]
    assert w[0]["start"] == 0.0 and w[0]["end"] == 0.4 and w[1]["end"] == 0.85


class _Res:
    def __init__(self, d):
        self.data = d


class _Llm:
    def complete_json(self, task, system, prompt, schema):
        assert task == "ai_style_rule"
        return _Res({"rules": ["Cut every pause longer than 0.4 s", "Open with the strongest sentence as a hook"]})


class _Eng:
    def creative_router(self):
        return _Llm()


@pytest.fixture
def projects(tmp_path):
    cfg = load_config()
    cfg.root = tmp_path
    (tmp_path / "drive").mkdir()
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "drive" / "AI videos"), "local": True, "data_dir": "ai"}
    return AV.Projects(cfg, Store(tmp_path / "jobs"), _Eng())


def test_feedback_becomes_rules_that_the_editor_brief_includes(projects):
    p = projects.create("My talk", "Open with the cost line", ["a.mov", "b.mov"], {}, {"clips": 0})
    assert projects.learn("too slow in the middle, and open stronger", "My talk")
    assert projects.learn("same again", "My talk") == []                      # duplicates are not added twice
    brief = projects.brief(projects.get(p["id"]))
    assert "Open with the cost line" in brief and "Cut every pause longer than 0.4 s" in brief
    job = projects.store.get(p["job"])
    assert job["status"] == "uploading" and job["options"]["mode"] == "long" and job["options"]["project"] and job["options"]["fcp"]


def test_make_sets_the_job_up_and_rejects_non_video_files(projects):
    with pytest.raises(ValueError):
        projects.create("x", "", ["notes.txt"], {}, {})
    p = projects.create("My talk", "n", ["a.mov"], {}, {"clips": 0})
    with pytest.raises(ValueError):
        projects.make(p["id"], {})                                                # still uploading
    projects.store.update(p["job"], status="draft")
    projects.make(p["id"], {"notes": "keep the demo", "music": "auto", "long_minutes": 8, "caption_style": "bold"})
    job = projects.store.get(p["job"])
    assert job["status"] == "queued" and job["options"]["long_minutes"] == 8.0 and "keep the demo" in job["options"]["prompt"]
    assert job["options"]["music"] == "auto" and job["options"]["caption_style"] == "bold"


def test_finished_video_is_archived_with_every_part_of_the_project(projects, tmp_path):
    p = projects.create("My: talk?", "the notes", ["a.mov"], {}, {"clips": 0})
    d = projects.store.dir(p["job"])
    (d / "source.mp4").write_bytes(b"raw")
    (d / "clip_01.mp4").write_bytes(b"video")
    (d / "clip_01.srt").write_text("1\n")
    (d / "plan.json").write_text(json.dumps({"intervals": [[0, 5], [8, 12]]}))
    (d / "transcript.json").write_text(json.dumps({"words": [{"text": "hello", "start": 0, "end": 1}]}))
    bundle = d / "fcp" / "My talk"
    bundle.mkdir(parents=True)
    (bundle / "My talk.fcpxml").write_text("<fcpxml/>")
    clip = {"index": 1, "file": "clip_01.mp4", "srt": "clip_01.srt", "title": "My talk", "rendered_at": 111, "duration": 9.0,
            "fcp": {"folder": "My talk"}, "story": {"summary": "a story", "files": [{"name": "a.mov", "kept_s": 9}]}}
    projects.store.update(p["job"], status="done", clips=[clip])
    assert projects.pending() == []                                              # no version recorded yet
    assert projects.archive(p["id"]) == "done"
    proj = projects.get(p["id"])
    dest = tmp_path / "drive" / "AI videos"
    folder = next(dest.iterdir())
    assert proj["version"] == 1 and proj["archive"]["state"] == "done"
    names = {str(x.relative_to(folder)) for x in folder.rglob("*") if x.is_file()}
    assert {"Raw footage/source.mp4", "Output/My talk v1.mp4", "Output/My talk v1.srt", "Final Cut/My talk/My talk.fcpxml",
            "Notes/brief.txt", "Notes/plan.md", "Notes/transcript.txt"} <= names
    assert "0.0 - 5.0" in (folder / "Notes" / "plan.md").read_text()
    # a second render is added as v2; the first video and the raw footage stay where they are
    projects.store.update(p["job"], clips=[{**clip, "rendered_at": 222, "edited": {"instruction": "trim the intro"}}])
    (d / "clip_01.mp4").write_bytes(b"video 2")
    assert projects.archive(p["id"]) == "done"
    assert (folder / "Output" / "My talk v1.mp4").read_bytes() == b"video" and (folder / "Output" / "My talk v2.mp4").read_bytes() == b"video 2"


def test_archive_waits_when_the_drive_is_not_mounted(projects, tmp_path):
    projects.cfg.data["clipper"]["ai_videos"]["dir"] = "/Volumes/No Such Drive/Videos/AI videos"
    p = projects.create("t", "", ["a.mov"], {}, {})
    d = projects.store.dir(p["job"])
    (d / "clip_01.mp4").write_bytes(b"v")
    projects.store.update(p["job"], status="done", clips=[{"index": 1, "file": "clip_01.mp4", "rendered_at": 1}])
    assert projects.archive(p["id"]) == "waiting"
    assert projects.get(p["id"])["archive"]["state"] == "waiting"
    from pathlib import Path
    assert not Path("/Volumes/No Such Drive").exists()                          # never creates a fake mount point


def test_title_card_html_is_sized_for_the_frame_and_escaped():
    from clipfarm.clipper import graphics as G
    h = G.title_html("5 <tips> & tricks", 3.2, 1080, 1920)
    assert "5 &lt;tips&gt; &amp; tricks" in h and 'data-width="1080"' in h and 'data-height="1920"' in h
    assert "window.__timelines" in h and 'data-duration="3.2"' in h
    wide = G.title_html("A much longer title that needs a smaller font to fit on two lines", 3.2, 1920, 1080)
    assert 'data-width="1920"' in wide
    assert G.render_title(load_config(), "", 3.0, 1080, 1920, __import__("pathlib").Path("/nonexistent/x.mov")) is None


def test_whole_video_edit_is_a_proper_method():
    import inspect

    from clipfarm.clipper.engine import Engine
    assert list(inspect.signature(Engine.edit_whole_video).parameters)[:3] == ["self", "job_id", "instruction"]


# ----------------------------------------------------------------------------- B-roll
def test_broll_scenes_are_validated_and_cutaways_only_go_where_words_match():
    from clipfarm.clipper import broll as B
    from clipfarm.clipper import moments as M
    scenes = B._valid_scenes([{"start_s": -3, "end_s": 6, "what": "a lake at dawn", "quality": 12},
                              {"start_s": 7, "end_s": 7.2, "what": "too short", "quality": 5}, {"start_s": 1}], 0.0, 5.0)
    assert scenes == [{"start": 0.0, "end": 5.0, "what": "a lake at dawn", "quality": 10.0}]
    sents = [M.Sentence(0, 0.0, 3.0, "Hi."), M.Sentence(1, 3.0, 9.0, "We drove to the lake.")]
    pool = [{"start": 20.0, "end": 26.0, "what": "a lake", "quality": 8.0}, {"start": 30.0, "end": 31.0, "what": "tiny", "quality": 9.0}]

    class Llm:
        def complete_json(self, task, system, prompt, schema, **kw):
            return _Res({"cutaways": [{"at": 1.0, "scene": 0, "length": 3},          # inside the hook: refused
                                      {"at": 4.0, "scene": 0, "length": 9},          # too long: trimmed to the maximum
                                      {"at": 5.0, "scene": 0, "length": 2},          # scene already used
                                      {"at": 50.0, "scene": 1, "length": 2}]})       # not during any sentence
    out = B.plan_cutaways(Llm(), sents, pool, "", 60.0)
    assert len(out) == 1 and out[0]["at"] == 4.0 and out[0]["length"] == B.CUT_MAX and out[0]["src"] == 20.0


def test_broll_named_files_are_the_pool_only_when_something_else_carries_the_story():
    from clipfarm.clipper import broll as B
    files = [{"name": "talk.mov", "start": 0, "end": 10}, {"name": "B-roll lake.mov", "start": 10, "end": 20},
             {"name": "walk.mov", "start": 20, "end": 30}]
    words = [{"text": "w", "start": i * 0.4, "end": i * 0.4 + 0.2} for i in range(25)]
    assert B.visual_files(files, words) == [1, 2]                                # the named one and the one with no speech
    assert B.pool_files(files, {"B-roll lake.mov": [{"what": "x"}]}, words) == {1}
    assert B.pool_files(files[1:2], {"B-roll lake.mov": [{"what": "x"}]}, words) == set()


def test_cutaway_is_rendered_over_the_talking_and_the_broll_file_is_not_in_the_story(tmp_path):
    import subprocess

    from PIL import Image

    from clipfarm.clipper import polish as P
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    cfg = load_config()
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    store = Store(tmp_path / "c")
    ff = ffmpeg.bin_path("ffmpeg")

    class Vision:
        def complete_json(self, task, system, prompt, schema, **kw):
            assert task == "broll_describe" and kw.get("images")
            return _Res({"scenes": [{"start_s": 12, "end_s": 20, "what": "a solid red wall", "quality": 9}]})

    class Writer:
        def complete_json(self, task, system, prompt, schema, **kw):
            if task == "broll_plan":
                return _Res({"cutaways": [{"at": 6.0, "scene": 0, "length": 2.0, "why": "mentions the red wall"}]})
            import re
            idx = [int(m) for m in re.findall(r"^\s+\[(\d+)\]", prompt, re.M)]
            return _Res({"title": "Talk", "keep": [{"first": idx[0], "last": idx[-1]}], "summary": "all of it"})

    class Eng(Engine):
        def vision_router(self):
            return Vision(), "fake"

        def creative_router(self):
            return Writer()
    eng = Eng(cfg, store)
    job = store.create({"files": ["talk.mov", "B-roll wall.mov"]}, {"mode": "long", "project": True, "caption_style": "bold",
                                                                     "pace": "natural", "fcp": True, "graphics": "title",
                                                                     "prompt": "show the wall"})
    d = store.dir(job["id"])
    src = d / "source.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=12", "-f", "lavfi",
                    "-i", "color=c=red:size=320x240:rate=30:duration=8", "-f", "lavfi", "-i", "sine=frequency=300:duration=20",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset",
                    "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True, timeout=120)
    store.update(job["id"], info={"files": [{"name": "talk.mov", "start": 0.0, "end": 12.0},
                                            {"name": "B-roll wall.mov", "start": 12.0, "end": 20.0}]})
    words = [{"text": f"w{i}" + ("." if i % 6 == 5 else ""), "start": 0.2 + i * 0.45, "end": 0.2 + i * 0.45 + 0.3} for i in range(25)]
    clip = P.run(eng, job["id"], src, words)
    assert len(clip["cutaways"]) == 1 and clip["duration"] < 13.5                 # the 8 s of B-roll are NOT appended to the story
    out = d / "clip_01.mp4"

    def corner(t):
        png = tmp_path / f"f{t}.png"
        subprocess.run([ff, "-y", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", str(png)], check=True)
        return Image.open(png).convert("RGB").getpixel((8, 8))
    r, g, b = corner(6.8)
    assert r > 200 and g < 60 and b < 60                                          # during the cutaway the picture is the red wall
    r2, g2, b2 = corner(3.0)
    assert not (r2 > 200 and g2 < 60 and b2 < 60)                                 # outside it, the speaker's footage
    assert json.loads((d / "plan.json").read_text())["cutaways"][0]["what"] == "a solid red wall"


# ----------------------------------------------------------------------------- the folder client
def _client():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("ai_edit", Path(__file__).resolve().parent.parent / "scripts" / "ai_edit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_folder_scan_orders_by_shooting_time_skips_results_and_marks_broll(tmp_path):
    import os
    c = _client()
    (tmp_path / "B-roll").mkdir()
    (tmp_path / "Output").mkdir()
    (tmp_path / "Final Cut").mkdir()
    for rel, t in [("talk2.mov", 200), ("talk1.mov", 100), ("B-roll/lake.mov", 150), ("Output/old v1.mp4", 50),
                   ("Final Cut/x/Media/source.mp4", 60), (".hidden.mp4", 70), ("notes.md", 10), ("sub/talk1.mov", 120)]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        os.utime(p, (t, t))
    names = [n for n, _ in c.scan(tmp_path)]
    assert names == ["talk1.mov", "talk1 (2).mov", "B-roll - lake.mov", "talk2.mov"]
    (tmp_path / "notes.md").write_text("Open with the cost line.")
    assert c.read_notes(tmp_path) == "Open with the cost line."


def test_server_gives_the_folder_client_everything_it_needs(tmp_path):
    """Full loop over HTTP against the real handler: create -> upload -> (job queued) -> plan endpoint."""
    import threading
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    cfg = load_config()
    cfg.root = tmp_path
    (tmp_path / "drive").mkdir()
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "drive" / "AI videos"), "local": True, "data_dir": "ai"}
    store = Store(tmp_path / "jobs")
    eng = Engine(cfg, store)
    feed = Feed(cfg, store)
    projects = AV.Projects(cfg, store, eng)
    worker = S.Worker(eng, store, feed, projects)            # not started: nothing will render
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, store, worker, feed, 100, projects))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        c = _client()
        api = c.Api(f"http://127.0.0.1:{httpd.server_address[1]}")
        proj = api.call("POST", "/api/ai/projects", {"title": "Trip", "notes": "keep the beach", "names": ["a.mov", "B-roll - b.mov"]})
        f = tmp_path / "a.mov"
        f.write_bytes(b"not really a video")
        api.upload(f"/api/upload?job={proj['job']}&index=0&name=a.mov", f)
        api.upload(f"/api/upload?job={proj['job']}&index=1&name=B-roll%20-%20b.mov", f)
        listed = api.call("GET", "/api/ai/projects")["projects"]
        assert listed[0]["job_state"]["status"] == "draft" and "keep the beach" in listed[0]["notes"]
        made = api.call("POST", f"/api/ai/projects/{proj['id']}/make", {"notes": "keep the beach", "music": "none", "graphics": "title"})
        assert made["job_state"]["status"] == "queued" and store.get(proj["job"])["options"]["graphics"] == "title"
        assert all(j["id"] != proj["job"] for j in api.call("GET", "/api/jobs"))
        api.call("DELETE", f"/api/ai/projects/{proj['id']}")
        assert api.call("GET", "/api/ai/projects")["projects"] == []
    finally:
        httpd.shutdown()


def test_letterboxed_broll_is_zoomed_to_fill_the_frame(tmp_path):
    import subprocess

    from clipfarm.clipper import broll as B
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    boxed = tmp_path / "boxed.mp4"                                              # a red picture with black bars top and bottom
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:size=320x120:rate=30:duration=3", "-vf",
                    "pad=320:240:0:60:black", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(boxed)], check=True)
    f = B.fill_filter(boxed, 0.0, 320, 240)
    assert f.startswith("crop=") and "force_original_aspect_ratio=increase" in f
    full = tmp_path / "full.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(full)], check=True)
    assert B.fill_filter(full, 0.0, 320, 240) == "scale=320:240"                # nothing to remove: just scale


def _live(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    cfg = load_config()
    cfg.root = tmp_path
    (tmp_path / "drive").mkdir(exist_ok=True)
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "drive" / "AI videos"), "local": True, "data_dir": "ai"}
    store = Store(tmp_path / "jobs")
    eng = Engine(cfg, store)
    projects = AV.Projects(cfg, store, eng)
    worker = S.Worker(eng, store, Feed(cfg, store), projects)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, store, worker, worker.feed, 100, projects))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, store, projects, _client().Api(f"http://127.0.0.1:{httpd.server_address[1]}")


def test_upload_resumes_after_the_server_restarts_and_only_missing_files_are_sent(tmp_path):
    c = _client()
    httpd, store, projects, api = _live(tmp_path)
    try:
        a, b, d = (tmp_path / n for n in ("a.mov", "b.mov", "c.mov"))
        a.write_bytes(b"A" * 1000)
        b.write_bytes(b"B" * 2000)
        d.write_bytes(b"C" * 3000)
        proj = api.call("POST", "/api/ai/projects", {"title": "Big", "names": ["a.mov", "b.mov", "c.mov"]})
        api.upload(f"/api/upload?job={proj['job']}&index=0&name=a.mov", a)                 # file 1 arrives, then "the Mac sleeps"
        store.requeue_interrupted()                                                          # (what a server restart does)
        job = store.get(proj["job"])
        assert job["status"] == "failed" and job["stage"] == "Upload paused"
        up = api.call("GET", f"/api/ai/projects/{proj['id']}/uploads")
        assert [f["size"] for f in up["files"]] == [1000, None, None]                      # the first file was kept
        sent = []
        real = api.upload_retry
        api.upload_retry = lambda path, f, **kw: (sent.append(f.name), real(path, f, **kw))[1]
        state = {"project": proj["id"], "job": proj["job"]}
        c.send_files(api, state, {"a.mov": a, "b.mov": b, "c.mov": d})
        assert sent == ["b.mov", "c.mov"]                                                    # a.mov was not sent again
        assert store.get(proj["job"])["status"] == "draft"
        assert [f["size"] for f in api.call("GET", f"/api/ai/projects/{proj['id']}/uploads")["files"]] == [1000, 2000, 3000]
    finally:
        httpd.shutdown()


def test_the_video_starts_by_itself_when_the_last_file_arrives_if_asked(tmp_path):
    httpd, store, projects, api = _live(tmp_path)
    try:
        f = tmp_path / "x.mov"
        f.write_bytes(b"x" * 500)
        proj = api.call("POST", "/api/ai/projects", {"title": "Auto", "notes": "n", "names": ["x.mov", "y.mov"],
                                                     "auto_make": {"notes": "keep the beach", "music": "none", "graphics": "title"}})
        api.upload(f"/api/upload?job={proj['job']}&index=0&name=x.mov", f)
        assert store.get(proj["job"])["status"] == "uploading"                               # one of two: not yet
        api.upload(f"/api/upload?job={proj['job']}&index=1&name=y.mov", f)
        job = store.get(proj["job"])
        assert job["status"] == "queued" and "keep the beach" in job["options"]["prompt"] and job["options"]["graphics"] == "title"
        assert "auto_make" not in projects.get(proj["id"]) or not projects.get(proj["id"])["auto_make"]
    finally:
        httpd.shutdown()


def test_a_half_received_file_is_not_counted_as_arrived(tmp_path):
    httpd, store, projects, api = _live(tmp_path)
    try:
        proj = api.call("POST", "/api/ai/projects", {"title": "Half", "names": ["a.mov"]})
        pd = store.dir(proj["job"]) / "parts"
        pd.mkdir(exist_ok=True)
        (pd / "part_000__a.mov.tmp").write_bytes(b"partial")
        assert api.call("GET", f"/api/ai/projects/{proj['id']}/uploads")["files"][0]["size"] is None
    finally:
        httpd.shutdown()


# ----------------------------------------------------------------------------- local projects (in place, OpenAI)
class _FakeAI:
    """Stands in for LocalAI: scripted answers, no network."""
    usage = {}

    def __init__(self, cfg=None, host=None):
        self.calls = []

    def available(self):
        return True, "fake"

    def cost(self):
        return 0.01

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        import re
        self.calls.append(task)
        idx = [int(m) for m in re.findall(r"^\s+\[(\d+)\]", prompt, re.M)]
        if task == "broll_describe":
            return _Res({"scenes": [{"start_s": 0, "end_s": 6, "what": "a solid red wall", "quality": 9}]})
        if task == "story_edit":
            return _Res({"title": "My trip", "keep": [{"first": idx[0], "last": idx[-1]}], "summary": "the whole thing"})
        if task == "hook_text":
            return _Res({"text": "You will not believe this"})
        if task == "broll_plan":
            return _Res({"cutaways": [{"at": 4.0, "scene": 0, "length": 2.0, "why": "the red wall"}]})
        if task == "speaker_roles":
            return _Res({"host": [], "has_interviewer": False})
        if task == "video_length":
            return _Res({"seconds": 0})
        if task == "key_points":
            return _Res({"points": []})
        if task == "music_vibe":
            return _Res({"mood": "chill", "genre": "", "why": "x"})
        if task == "revise_request":
            return _Res({"shorter_pct": 25, "shorter_by_s": 0, "longer_by_s": 0, "target_total_s": 0, "change_music": True,
                         "music_request": "", "beat_sync": "on", "hook_text": "WATCH THIS FIRST", "better_hook": False,
                         "move_to_front_quote": "", "content_instruction": "", "summary": "shorter, new song on the beat, new hook"})
        if task == "clip_edit":                                   # keep only the first 70% of what is shown
            k = max(1, int(len(idx) * 0.7))
            return _Res({"segments": [{"first": idx[0], "last": idx[k - 1]}], "title": "", "summary": "dropped the weakest ends", "problem": ""})
        if task == "video_kind":                                  # what the notes ask for
            return _Res({"kind": "promo" if "punchy" in prompt.lower() else "story"})
        if task == "ai_style_rule":                               # a change asked for on one video teaches nothing general here
            return _Res({"rules": []})
        if task == "flow_recheck":
            return _Res({"steps": []})
        raise AssertionError(f"unexpected AI task {task}")

    def router_for_now(self):
        return self, "fake"

    def vision_router(self):
        return self, "fake"

    def creative_router(self):
        return self


def _make_folder(tmp_path):
    import subprocess

    import numpy as np

    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    f = tmp_path / "My trip"
    (f / "B-roll").mkdir(parents=True)
    (f / "Music").mkdir()
    for name, d in (("talk1.mov", 14), ("talk2.mov", 10)):
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30:duration={d}", "-f", "lavfi",
                        "-i", f"sine=frequency=300:duration={d}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(f / name)], check=True, timeout=120)
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:size=480x270:rate=30:duration=6", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(f / "B-roll" / "red wall.mov")], check=True, timeout=120)
    for n in ("a", "b"):                                                                 # two 120 BPM songs to switch between
        sr = 22050
        y = np.zeros(sr * 40, dtype=np.float32)
        t = 0.35
        while t < 39.8:
            i = int(t * sr)
            m = int(0.03 * sr)
            y[i:i + m] += (np.sin(2 * np.pi * (900 if n == "a" else 700) * np.arange(m) / sr) * np.exp(-np.arange(m) / (0.006 * sr))).astype(np.float32)
            t += 0.5
        raw = tmp_path / f"{n}.f32"
        raw.write_bytes(y.tobytes())
        subprocess.run([ff, "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(raw), str(f / "Music" / f"song {n}.wav")], check=True)
    (f / "notes.md").write_text("Make it punchy. Use the red wall cutaway.")
    return f


def test_local_project_makes_a_video_in_place_then_applies_a_several_part_change(tmp_path, monkeypatch):
    import os

    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True}
    folder = _make_folder(tmp_path)
    before = {p: (p.stat().st_size, p.stat().st_mtime) for p in folder.rglob("*.mov")}
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    fake = _FakeAI()
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)
    words = {"talk1.mov": [{"text": f"w{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(24)],
             "talk2.mov": [{"text": f"v{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(16)]}
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words[f["name"]])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": "auto"}, {"clips": 0})      # (music only when asked for: see the next test)
    assert proj["mode"] == "local" and "punchy" in proj["notes"]
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    out = sorted((folder / "Output").glob("*.mp4"))
    assert [p.name for p in out] == ["My trip v1.mp4"] and (folder / "Notes" / "plan.md").exists()
    assert any((folder / "Final Cut").rglob("*.fcpxml"))
    assert "B-roll cutaways" in (folder / "Notes" / "plan.md").read_text() and "red wall" in (folder / "Notes" / "plan.md").read_text()
    v1_len = ffmpeg.duration(out[0])
    assert 15 < v1_len < 26                                                            # B-roll is a cutaway, not appended
    plan1 = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan1["hook_text"] == "You will not believe this" and plan1["music"]["file"].endswith(".wav")
    assert plan1["style"] == "promo" and not plan1.get("pending_render")                # "punchy" notes = a promo; rendered, nothing pending
    from clipfarm.clipper import music as MU
    assert -13.5 < MU.integrated(out[0]) < -10.5                                       # a promo finishes louder (about -12 LUFS)
    import subprocess
    rough = next((folder / ".aivideo").glob("rough_*.mov"))                             # the rough cut has every frame: no holes at the cuts
    pts = [float(x) for x in subprocess.run([ffmpeg.bin_path("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                                            "-of", "csv=p=0", str(rough)], capture_output=True, text=True).stdout.split()]
    pts.sort()
    assert pts and max(b - a for a, b in zip(pts, pts[1:])) < 1.5 / 30
    assert (folder / ".aivideo" / "plan_words.json").exists()                           # the words this edit was made of are kept with it
    # one message, several changes
    runner.revise(proj["id"], "slightly too long, cut it shorter. bend to the beat, switch the song, add a better hook that says WATCH THIS FIRST")
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    plan2 = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan2["hook_text"] == "WATCH THIS FIRST" and plan2["beat_sync"] is True
    assert plan2["music"]["file"] != plan1["music"]["file"]                            # a different song
    out2 = sorted((folder / "Output").glob("*.mp4"))
    assert [p.name for p in out2] == ["My trip v1.mp4", "My trip v2.mp4"]
    assert ffmpeg.duration(folder / "Output" / "My trip v2.mp4") < v1_len - 2         # shorter
    assert "ai_style_rule" in fake.calls                                               # the change was offered to the rule-learner, on THIS AI
    assert len([d for d in (folder / "Final Cut").iterdir() if d.is_dir()]) == 2       # v1's Final Cut folder was kept, v2 has its own
    meta = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    cuts = [sum(b - a for a, b in meta["intervals"][:i + 1]) for i in range(len(meta["intervals"]) - 1)]
    assert store.get(proj["job"])["clips"][0]["beat"].startswith("120 BPM") or store.get(proj["job"])["clips"][0]["beat"].startswith("1")
    on_beat = [abs(c / 0.5 - round(c / 0.5)) < 0.12 for c in cuts]
    assert not cuts or any(on_beat)
    # the footage was only read
    assert before == {p: (p.stat().st_size, p.stat().st_mtime) for p in before}


def test_local_project_looks_at_silent_footage_picks_shots_and_keeps_the_rest_as_cutaways(tmp_path, monkeypatch):
    import subprocess

    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    f = tmp_path / "Shoes"
    f.mkdir()
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=12", "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(f / "talk.mov")], check=True, timeout=120)
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=30:duration=14", "-c:v", "libx264", "-preset",
                    "ultrafast", "-pix_fmt", "yuv420p", str(f / "shop shelves.mov")], check=True, timeout=120)            # no audio, not named B-roll
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    seen = {}

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_describe":
                seen["described"] = prompt
                return _Res({"scenes": [{"start_s": 0, "end_s": 6, "what": "shoe shelves in a shop", "quality": 9},
                                        {"start_s": 8, "end_s": 13, "what": "a customer trying shoes", "quality": 8}]})
            if task == "visual_pick":
                seen["pick"] = prompt
                return _Res({"keep": [0], "why": "the shelves introduce the shop"})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    monkeypatch.setattr(LP, "LocalAI", lambda c: AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, fl, ai: [
        {"text": f"w{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(22)])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(f), "", "Show the shop", {"transcriber": "openai"}, {"clips": 0})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done", store.get(proj["job"])
    assert "shop shelves.mov" in seen["described"]                                       # the silent file was LOOKED AT
    assert "shoe shelves in a shop" in seen["pick"]                                      # and the AI chose among what it saw
    plan = json.loads((f / ".aivideo" / "plan.json").read_text())
    kinds = [it["kind"] for it in plan["items"]]
    assert "visual" in kinds and kinds.count("visual") == 1
    assert [s["what"] for s in plan["pool"]] == ["a customer trying shoes"]            # the shot that wasn't played is a cutaway option
    length = ffmpeg.duration(next((f / "Output").glob("*.mp4")))
    assert 17 < length < 22                                                              # ~12 s of talk + the 6 s shelves shot (max 8)


def test_director_chatter_around_a_shot_does_not_make_it_talking_footage(tmp_path):
    from clipfarm.clipper import localproj as LP
    f = {"name": "x.mov", "start": 0.0, "end": 20.0, "dur": 20.0, "audio": True}
    chatter = [{"text": t, "start": a, "end": a + 0.4} for t, a in [("go", 0.5), ("that", 18.0), ("was", 18.4), ("good", 18.8), ("cut", 19.3)]]
    assert LP.LocalRunner.sparse(f, chatter) is True                       # 5 short words in 20 s of picture
    talking = [{"text": "w", "start": i * 0.5, "end": i * 0.5 + 0.35} for i in range(36)]
    assert LP.LocalRunner.sparse(f, talking) is False
    assert LP.LocalRunner.sparse({**f, "audio": False}, []) is True


def test_multicam_shoot_is_synced_directed_cut_from_the_originals_and_exported_as_a_final_cut_multicam(tmp_path, monkeypatch):
    import subprocess

    import numpy as np
    from PIL import Image

    from clipfarm.clipper import fcpxml as FX
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    SR = 16000
    rng = np.random.default_rng(4)
    voice = np.zeros(SR * 70, dtype=np.float32)
    t = 0.0
    while t < 69:
        d = rng.uniform(0.2, 0.8)
        i, j = int(t * SR), int(min(70, t + d) * SR)
        voice[i:j] = rng.standard_normal(j - i).astype(np.float32) * rng.uniform(0.2, 0.6)
        t += d + rng.uniform(0.05, 0.4)

    def wav(path, y):
        raw = path.with_suffix(".f32")
        raw.write_bytes(y.astype(np.float32).tobytes())
        subprocess.run([ff, "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw), str(path)], check=True)

    def cam(path, color, start, dur):
        y = voice[int(start * SR):int((start + dur) * SR)]
        y = np.convolve(y, np.ones(5) / 5, mode="same") + rng.standard_normal(len(y)).astype(np.float32) * 0.04
        w = path.with_suffix(".wav")
        wav(w, y)
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:size=320x180:rate=24:duration={dur}", "-i", str(w), "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True, timeout=120)
        w.unlink()
    folder = tmp_path / "Interview"
    for d in ("Cam A", "Cam B", "Audio"):
        (folder / d).mkdir(parents=True)
    cam(folder / "Cam A" / "C0001.mp4", "red", 0.0, 70)
    cam(folder / "Cam B" / "C0001.mp4", "blue", 10.0, 50)                 # B started 10 s in and stopped 10 s early
    wav(folder / "Audio" / "ZOOM0001.wav", voice)                          # the recorder's clean sound
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False,
                                        "hyperframes_dir": "/nonexistent"}
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "multicam_director":
                assert images, "the director must be shown frames of every angle"
                return _Res({"segments": [{"start": 0, "end": 22, "angle": "Cam A", "why": "wide"}, {"start": 22, "end": 45, "angle": "Cam B", "why": "close-up"},
                                          {"start": 45, "end": 70, "angle": "Cam A", "why": "wide"}]})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    monkeypatch.setattr(LP, "LocalAI", lambda c: AI())
    words_rec = [{"text": f"w{i}" + ("." if i % 6 == 5 else ""), "start": 1.0 + i * 0.7, "end": 1.0 + i * 0.7 + 0.4} for i in range(90)]
    seen = []
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: (seen.append(f["path"]), words_rec)[1])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "", "", {"transcriber": "openai"}, {"clips": 0})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    assert seen == [str(folder / "Audio" / "ZOOM0001.wav")]                            # only the recorder was transcribed, once
    grp = json.loads((folder / ".aivideo" / "multicam.json").read_text())["group"]
    assert [a["id"] for a in grp["angles"]] == ["Cam A", "Cam B"] and abs(grp["angles"][1]["files"][0]["t0"] - 10.0) < 0.06
    clip = job["clips"][0]
    assert clip["multicam"]["angles"] == ["Cam A", "Cam B"] and clip["fcp"]["multicam"] is True
    # the picture really switches: red (Cam A) at the start, blue (Cam B) in the middle of its stretch
    meta = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    iv = meta["intervals"]

    def out_time(master):
        acc = 0.0
        for a, b in iv:
            if a <= master <= b:
                return acc + (master - a)
            acc += b - a
        return None
    out = next((folder / "Output").glob("*.mp4"))

    def pixel(master):
        tt = out_time(master)
        png = tmp_path / f"p{master}.png"
        subprocess.run([ff, "-y", "-v", "error", "-ss", f"{tt:.2f}", "-i", str(out), "-frames:v", "1", str(png)], check=True)
        return Image.open(png).convert("RGB").getpixel((6, 6))
    r, g, b = pixel(15.0)
    assert r > 150 and b < 100
    r, g, b = pixel(33.0)
    assert b > 150 and r < 100
    # Final Cut gets a real multicam clip pointing at the ORIGINAL files, valid for Final Cut itself
    xmls = list((folder / "Final Cut").rglob("*.fcpxml"))
    assert xmls
    text = xmls[0].read_text()
    assert "<multicam" in text and "<mc-clip" in text and "Cam B" in text and "Interview/Cam B/C0001.mp4" in text.replace("%20", " ")
    assert not (xmls[0].parent / "Media").exists()
    if FX.DTD_DIR.exists():
        assert FX.validate(xmls[0]) == []


def test_the_interviewer_is_taken_out_so_only_the_guest_can_be_used(tmp_path):
    from clipfarm.clipper import speakers as SP
    words = []
    t = 0.0
    script = [("Tell me about your shoe store?", "host"), ("We fit every customer by hand.", "guest"), ("Why does that matter?", "host"),
              ("Because the right shoe changes how you move.", "guest")]
    for text, who in script:
        for i, w in enumerate(text.split()):
            words.append({"text": w, "start": t, "end": t + 0.3, "who": who})
            t += 0.35
        t += 0.8

    class AI:
        def complete_json(self, task, system, prompt, schema, **kw):
            import re
            host = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\] [^\n]*\?\s*$", prompt, re.M)]       # 'a question mark = the interviewer'
            return _Res({"host": host, "has_interviewer": bool(host)})
    spans = SP.host_spans(AI(), [{k: v for k, v in w.items() if k != "who"} for w in words])
    kept = SP.drop_spans([{k: v for k, v in w.items() if k != "who"} for w in words], spans)
    assert len(spans) == 2
    texts = " ".join(w["text"] for w in kept)
    assert "fit every customer" in texts and "right shoe" in texts and "Tell me" not in texts and "Why does" not in texts


def test_a_video_that_freezes_or_has_mismatched_sound_is_not_delivered(tmp_path):
    import subprocess

    from clipfarm.clipper import localproj as LP
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    good = tmp_path / "good.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=12", "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(good)], check=True, timeout=120)
    assert LP.check_video(good, 12.0) == []
    # the picture stops at 6 s but the sound plays on: exactly what a half-finished render looks like
    frozen = tmp_path / "frozen.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-i", str(good), "-vf", "tpad=stop_mode=clone:stop_duration=0,trim=0:6,tpad=stop_mode=clone:stop_duration=6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(frozen)], check=True, timeout=120)
    probs = LP.check_video(frozen, 12.0)
    assert any("freezes" in x for x in probs)
    # picture much shorter than the sound
    short = tmp_path / "short.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-i", str(good), "-t", "5", "-an", str(tmp_path / "v.mp4")], check=True, timeout=120)
    subprocess.run([ff, "-y", "-v", "error", "-i", str(tmp_path / "v.mp4"), "-i", str(good), "-map", "0:v", "-map", "1:a", "-c", "copy", str(short)], check=True, timeout=120)
    assert any("sound is" in x for x in LP.check_video(short, 12.0))
    assert any("should be" in x for x in LP.check_video(good, 30.0))


def test_slivers_are_joined_to_the_piece_before_or_dropped_and_pieces_are_whole_frames():
    from clipfarm.clipper import localproj as LP
    mk = lambda path, ss, d: {"d": d, "v": {"path": "v", "ss": 0, "fill": False}, "a": {"path": path, "ss": ss, "pre": 0}, "item": {"kind": "clip", "path": "v", "start": 0},   # noqa: E731
                              "out": 0}
    parts = [mk("a.wav", 10.0, 5.0), mk("a.wav", 15.0, 0.04), mk("b.wav", 3.0, 4.0), mk("c.wav", 0.0, 0.03), mk("b.wav", 7.0, 2.0)]
    out = LP.quantize(parts, 24.0)
    assert [round(p["d"] * 24) for p in out] == [121, 96, 48]                   # the 0.04 s joined the first piece; the 0.03 s was dropped
    assert all(abs(p["d"] * 24 - round(p["d"] * 24)) < 1e-6 for p in out)
    assert abs(sum(p["d"] for p in out) - 11.04) < 0.05


def test_a_quiet_video_is_brought_up_to_the_target_loudness(tmp_path):
    import re
    import subprocess

    from clipfarm.clipper import localproj as LP
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    v = tmp_path / "quiet.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=8", "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                    "-af", "volume=-30dB", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(v)], check=True, timeout=120)
    before = LP.loudness_pass(v, -12.0)
    assert before is not None and before < -30
    r = subprocess.run([ff, "-v", "info", "-i", str(v), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    after = float(re.findall(r"\bI:\s+(-?\d+(?:\.\d+)?)\s+LUFS", r.stderr)[-1])
    assert -13.5 < after < -10.5


def test_recorder_audio_is_found_beside_the_footage_or_where_you_point(tmp_path):
    from clipfarm.clipper import localproj as LP
    import subprocess
    from clipfarm.core import ffmpeg
    def wav(p):
        subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=8", str(p)], check=True)
    proj = tmp_path / "Shoe Shack"
    raw = proj / "Raw Footage"
    raw.mkdir(parents=True)
    other = tmp_path / "elsewhere"
    other.mkdir()
    wav(proj / "rec_tr1.wav")                  # just above the footage folder
    wav(raw / "inside.wav")
    wav(other / "lav.wav")
    names = sorted(a["name"] for a in LP.scan_audio(raw))
    assert names == ["inside.wav", "rec_tr1.wav"]
    assert sorted(a["name"] for a in LP.scan_audio(raw, [str(other / "lav.wav")])) == ["inside.wav", "lav.wav", "rec_tr1.wav"]
    assert sorted(a["name"] for a in LP.scan_audio(raw, [str(other)])) == ["inside.wav", "lav.wav", "rec_tr1.wav"]


def test_a_whole_project_folder_is_read_the_way_a_video_editor_lays_it_out(tmp_path):
    import subprocess

    from clipfarm.clipper import localproj as LP
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    root = tmp_path / "Shoe Shack"
    for d in ("Raw Footage", "B-roll", "Exports", "Final Cut Proxy Media/2025-05-09", "Final Cut Optimized Media", "Shoe Shack.fcpbundle/5-2-25",
              "Sound Effects", "Shot Ideas", "Graphics", "Music"):
        (root / d).mkdir(parents=True)

    def vid(p):
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2", "-pix_fmt", "yuv420p", str(p)], check=True)

    def wav(p):
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=8", str(p)], check=True)
    for p in ("Raw Footage/C0005.MP4", "B-roll/store.mov", "Exports/old edit.mov", "Final Cut Proxy Media/2025-05-09/C0005.mov", "Final Cut Optimized Media/x.mov",
              "Shoe Shack.fcpbundle/5-2-25/render.mov", "Shot Ideas/IMG_1.mov"):
        vid(root / p)
    wav(root / "250509_001_Tr1.WAV")
    wav(root / "Sound Effects" / "whoosh.wav")
    wav(root / "Music" / "song.wav")
    for p in ("logo.png", "Graphics/brand.jpg", "Raw Footage/screengrab.png"):
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:size=64x64:duration=1", "-frames:v", "1", str(root / p)], check=True)
    names = sorted(f["name"] for f in LP.scan(root))
    assert names == ["C0005.MP4", "IMG_1.mov", "store.mov"]                 # not the proxies, the old export, or anything inside the Final Cut library
    assert [f["name"] for f in LP.scan(root) if f["broll"]] == ["store.mov"]  # the B-roll folder
    assert [a["name"] for a in LP.scan_audio(root)] == ["250509_001_Tr1.WAV"]  # the recorder - not sound effects, not the Music folder
    assert sorted(g["name"] for g in LP.scan_graphics(root)) == ["brand.jpg", "logo.png"]


def test_the_editor_starts_with_everything_it_was_taught_and_keeps_learning_from_changes(projects):
    rules = projects.rules()
    for must in ("ADVERTISEMENTS", "INTERVIEWS", "Never change the colour", "editable title", "-12 LUFS", "ALL its camera angles"):
        assert must in rules
    p = projects.create("t", "", ["a.mov"], {}, {})
    brief = projects.brief(projects.get(p["id"]))
    assert "mostly B-roll" in brief and "COMPLETE closing line" in brief            # the editor's brief for EVERY project carries them
    before = projects.rules()
    projects.add_rules(AV.DEFAULT_RULES)                                            # restarting adds nothing twice
    assert projects.rules() == before
    assert projects.learn("slightly too long, cut shorter", "t", fallback=False) != ["slightly too long, cut shorter"]   # never saves raw instructions


def test_many_graphics_are_joined_into_one_track_and_the_final_render_does_not_stall(tmp_path):
    """Regression: nine separately-timed overlay inputs made ffmpeg hang forever once one had ended and the next not begun."""
    import subprocess
    import time

    from clipfarm.clipper import polish as P
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    movs = []
    for i in range(9):
        m = tmp_path / f"g{i}.mov"
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red@0.6:s=320x180:r=24:d=2.5,format=rgba", "-c:v", "qtrle", "-pix_fmt", "argb", str(m)],
                       check=True, timeout=60)
        movs.append((6.0 + i * 8.0, 2.5, m))                                           # at 6 s, 14 s, ... 70 s
    track = P.overlay_track(movs, 80.0, 320, 180, 24.0, tmp_path / "ov.mov")
    assert track is not None and abs(ffmpeg.duration(track) - 80.0) < 0.2
    base = tmp_path / "base.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=80", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base)], check=True, timeout=120)
    out = tmp_path / "out.mp4"
    t = time.time()
    ffmpeg.run([ff, "-y", "-v", "error", "-i", str(base), "-i", str(track), "-filter_complex", "[1:v]format=rgba[ov];[0:v][ov]overlay=0:0:format=auto:eof_action=pass,format=yuv420p[v]",
                "-map", "[v]", "-c:v", "libx264", str(out)], timeout=120, what="overlay test")
    assert time.time() - t < 60 and abs(ffmpeg.duration(out) - 80.0) < 0.3
    # the transparent stretches stay transparent: the picture shows through at 1 s, the graphic at 7 s
    from PIL import Image

    def px(t_, at=(160, 90)):
        png = tmp_path / f"p{t_}.png"
        subprocess.run([ff, "-y", "-v", "error", "-ss", str(t_), "-i", str(out), "-frames:v", "1", str(png)], check=True)
        return Image.open(png).convert("RGB").getpixel(at)
    plain = px(1.0)
    with_gfx = px(7.0)
    assert with_gfx[0] > plain[0] + 20 or with_gfx[0] > 120                                  # red graphic over the picture


def test_the_ai_videos_key_is_read_from_env_used_for_calls_and_every_call_is_counted(tmp_path, monkeypatch, caplog):
    import json as _json

    import requests

    from clipfarm.clipper import localai as LAI
    secret = "sk-test-SECRET-1234567890"
    (tmp_path / ".env").write_text(f"OTHER=1\n{LAI.KEY_ENV}={secret}\n")
    cfg = load_config()                                   # (this loads the project's real .env into the environment: clear it AFTER)
    monkeypatch.delenv(LAI.KEY_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg.root = tmp_path
    seen = {}

    class R:
        status_code = 200
        text = ""

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

        def raise_for_status(self):
            pass

    def post(url, headers=None, json=None, **kw):
        seen["auth"] = (headers or {}).get("Authorization")
        if "transcriptions" in url:
            return R({"words": [{"word": "hi", "start": 0.0, "end": 0.3}]})
        return R({"model": "gpt-4.1-mini-x", "choices": [{"message": {"content": _json.dumps({"ok": True})}}], "usage": {"prompt_tokens": 2000, "completion_tokens": 500}})
    monkeypatch.setattr(requests, "post", post)
    ai = LAI.LocalAI(cfg)
    assert ai.available()[0] and "your AI Videos key" in ai.available()[1]
    res = ai.complete_json("t", "sys", "prompt", {"type": "object"})
    assert res.data == {"ok": True} and (seen["auth"] == f"Bearer {secret}") is True
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"x")
    assert ai.transcribe_mp3(mp3, 120.0)[0]["text"] == "hi"
    s = LAI.ledger_for(cfg).summary()
    assert s["yours_today"]["calls"] == 1 and s["yours_today"]["in"] == 2000 and s["yours_today"]["whisper_s"] == 120.0
    assert abs(s["yours_total"]["usd"] - (2000 * 0.4 / 1e6 + 500 * 1.6 / 1e6 + 120 / 60 * 0.006)) < 1e-4
    assert s["today"]["usd"] == s["yours_today"]["usd"]
    assert secret not in caplog.text and secret not in (tmp_path / "data" / "ai_videos" / "usage.json").read_text()   # never written anywhere but .env


def test_cuts_never_land_mid_word_and_the_interviewer_stays_out():
    import numpy as np

    from clipfarm.clipper import tighten as TI
    e = np.full(600, -60.0, dtype=np.float32)
    e[100:260] = -20.0                                         # one spoken stretch, 1.00 s to 2.60 s
    thr = TI._threshold(np.concatenate([e, np.full(600, -60.0, dtype=np.float32)]))
    assert abs(TI.end_of(e, thr, 2.4) - 2.60) < 0.03           # a cut inside the word runs on to its end
    assert abs(TI.end_of(e, thr, 2.9) - 2.60) < 0.03           # a cut in dead air is pulled back to the last sound
    assert abs(TI.start_of(e, thr, 0.4) - 1.00) < 0.03         # dead air before the first word is trimmed
    assert TI.without([(0.0, 10.0)], [(4.0, 5.0)]) == [(0.0, 3.82), (5.18, 10.0)]


def test_hedging_lines_and_photos_are_kept_out_of_an_ad():
    from clipfarm.clipper import localproj as LP
    assert LP.HEDGE_RE.search("also I'm not entirely sure how long we've been open")
    assert not LP.HEDGE_RE.search("we fit every foot by hand")


def test_no_moment_or_line_plays_twice_and_something_changes_every_twelve_seconds():
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper import moments as M
    assert LP.dedupe_intervals([(10.0, 20.0), (5.0, 12.0), (30.0, 31.0)]) == [(10.0, 20.0), (5.0, 10.0), (30.0, 31.0)]
    sents = [M.Sentence(0, 0.0, 3.0, "We fit every foot by hand here."), M.Sentence(1, 5.0, 8.0, "we fit every foot by hand here"),
             M.Sentence(2, 9.0, 12.0, "Come see us on Main Street today.")]
    items = [{"kind": "talk", "a": i, "b": i} for i in range(3)]
    assert [it["a"] for it in LP.no_repeats(items, sents)] == [0, 2]
    pool = [{"start": 100.0 + 10 * i, "end": 106.0 + 10 * i, "what": f"shoes {i}", "quality": 8, "file": f"f{i}"} for i in range(6)]
    cuts, out_len = [], 50.0
    c2, _ = LP.keep_moving([], cuts, [(0.0, 50.0)], pool, None, None, out_len)
    marks = sorted({0.0, out_len} | {c["at"] for c in c2} | {c["at"] + c["length"] for c in c2})
    assert all(b - a <= 12.1 for a, b in zip(marks, marks[1:])) and 2 <= len(c2) <= 5          # twelve seconds, not more cutting than that needs


def test_the_cleanest_camera_provides_all_the_sound(tmp_path):
    from clipfarm.clipper import multicam as MC
    a = {"path": "A.mov", "t0": 0.0, "dur": 100.0}
    b = {"path": "B.mov", "t0": 0.0, "dur": 90.0}
    group = {"angles": [{"id": "A", "files": [a]}, {"id": "B", "files": [b]}], "recorders": [], "sound_q": {"A.mov": 12.0, "B.mov": 31.0}}
    assert MC.sound_at(group, 10.0)["path"] == "B.mov" and MC.sound_at(group, 50.0)["path"] == "B.mov"
    assert MC.sound_at(group, 95.0)["path"] == "A.mov"                       # only A is recording there


def test_a_photo_becomes_a_short_animated_clip(tmp_path):
    import subprocess

    from clipfarm.clipper import photos as PH
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    pic = tmp_path / "p.jpg"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=800x600:rate=1", "-frames:v", "1", str(pic)], check=True)
    for effect in ("zoom_in", "polaroid", "film"):
        out = PH.bake(str(pic), effect, 1.5, 640, 360, 30.0, tmp_path / f"{effect}.mp4")
        assert abs(ffmpeg.duration(out) - 1.5) < 0.1


def test_the_outline_in_the_notes_is_the_script():
    from clipfarm.clipper import flow as FL
    notes = ("This should be the flow of the video:\n    1.    Unexpected pregnancy: 22, design school.\n    2.    The uncertainty: single motherhood.\n"
             "    3.    Harbor enters the story: Hannah, Little Steps,\n          other moms.\n    4.    Closing.\n\nuse a decent amount of b roll")
    got = FL.steps(notes)
    assert len(got) == 4 and got[0].startswith("Unexpected pregnancy") and "other moms" in got[2]
    assert FL.steps("just make it good\n1. one thing") == []


def test_two_identical_cameras_are_told_apart_by_their_sound(tmp_path):
    from clipfarm.clipper import multicam as MC
    a = {"A": [{"t0": 0.0, "dur": 100.0, "path": "a"}], "B": [{"t0": 50.0, "dur": 100.0, "path": "b"}]}
    one = {"Camera 1": [{"t0": 0.0, "dur": 100.0, "path": "a"}, {"t0": 50.0, "dur": 100.0, "path": "b"}]}
    out = MC.split_overlaps(one)
    assert sorted(out) == ["Camera 1", "Camera 2"] and out["Camera 2"][0]["path"] == "b"
    assert sorted(MC.split_overlaps(a)) == ["A", "B"]


def test_the_script_is_checked_word_for_word_for_names_numbers_and_quotes():
    from clipfarm.clipper import flow as FL
    outline = ["Unexpected pregnancy: 22, design school", "Harbor enters the story: Hannah Halden, Little Steps, other moms",
               'Deeper impact: "the love of Christ whenever you walked in the door"']
    must = FL.requirements(outline)
    assert must[1][0] == ("22", True) and must[1][1] == ("design school", False)
    assert ("Hannah Halden", True) in must[2] and ("other moms", False) in must[2]
    assert must[3][0][1] is True
    assert FL.heard("Hannah Halden", "Hannon She was the director") and not FL.heard("Hannah Halden", "the director of the program")
    assert FL.heard("22", "I was 22 years old") and FL.heard("22", "I was twenty two") and not FL.heard("22", "I was thirty")
    assert FL.heard("the love of Christ whenever you walked in the door", "you could feel the love of Christ whenever you walked in the door") \
        and not FL.heard("the love of Christ whenever you walked in the door", "the love of my family")


def test_a_bulleted_script_is_a_script_too():
    from clipfarm.clipper import flow as FL
    notes = ("You MUST lay the video out in this order.\n\n\xa0\xa0\xa0\xa0•\xa0\xa0\xa0\xa0Young couple with daughter, Lucia, moved from Hawaii to Dubuque.\n"
             "\xa0\xa0\xa0\xa0•\xa0\xa0\xa0\xa0The move left them without jobs.\n\xa0\xa0\xa0\xa0•\xa0\xa0\xa0\xa0WIC referred them to Harbor Clinic.\n"
             "\xa0\xa0\xa0\xa0•\xa0\xa0\xa0\xa0Community: people who were \"cheering for us.\"\n\nuse b roll")
    steps = FL.steps(notes)
    assert len(steps) == 4 and steps[0].startswith("Young couple") and steps[2].startswith("WIC")
    must = FL.requirements(steps)
    assert ("Lucia", True) in must[1] and ("Hawaii", True) in must[1] and ("WIC", True) in must[3] and ("Harbor Clinic", True) in must[3]
    assert ("cheering for us", True) in must[4]


def test_a_new_local_project_has_no_background_music_unless_asked(tmp_path):
    """The library is one well-known catalogue ("meme music") that ruins a video: no music is the default, 'auto' / a mood is an explicit choice."""
    from clipfarm.clipper.engine import Engine
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True}
    folder = _make_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    p = projects.create_local(str(folder), "", "", {"transcriber": "openai"}, {"clips": 0})
    assert store.get(p["job"])["options"]["music"] == ""
    folder2 = _make_folder(tmp_path / "b")
    p2 = projects.create_local(str(folder2), "", "", {"transcriber": "openai", "music": "chill"}, {"clips": 0})
    assert store.get(p2["job"])["options"]["music"] == "chill"


# ----------------------------------------------------------------------------- "Add B-roll to a video" (keep_cut)
def _keep_cut_folder(tmp_path):
    import subprocess

    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    f = tmp_path / "Open house"
    f.mkdir()
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=20", "-f", "lavfi", "-i", "sine=frequency=300:duration=20",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(f / "Interview.mov")], check=True, timeout=120)
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=green:size=320x240:rate=30:duration=8", "-f", "lavfi", "-i", "sine=frequency=500:duration=8",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(f / "Walk and talk.mov")], check=True, timeout=120)
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=30:duration=6", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(f / "Campus.mov")], check=True, timeout=120)
    return f


def test_add_broll_keeps_the_whole_video_once_and_lays_broll_over_it(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = _keep_cut_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_describe":
                return _Res({"scenes": [{"start_s": 0, "end_s": 5.5, "what": "students walking on campus", "quality": 8}]})
            if task == "broll_plan":
                return _Res({"cutaways": [{"at": 3.0, "scene": 0, "length": 2.5, "why": "campus"}, {"at": 13.0, "scene": 1, "length": 2.5, "why": "walk"}]})
            if task == "broll_pick":
                return _Res({"scene": 0, "why": "x"})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    fake = AI()
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)
    heard = []
    # the interview: an "um", a stutter and six seconds of silence in the middle (a normal edit would cut all of them)
    said = ["Welcome", "um", "to", "to", "the", "open", "house."] + ["We", "start", "with", "the", "labs."]
    words = [{"text": w, "start": 0.5 + i * 0.6, "end": 0.5 + i * 0.6 + 0.4} for i, w in enumerate(said[:7])] + \
            [{"text": w, "start": 12.0 + i * 0.6, "end": 12.0 + i * 0.6 + 0.4} for i, w in enumerate(said[7:])]

    def words_of(self, f, ai):
        heard.append(f["name"])
        return words
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", words_of)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Open house", "", {"transcriber": "openai"}, {"clips": 0})
    with pytest.raises(ValueError):
        projects.prepare_local(proj["id"], {"keep_cut": True, "aroll": "../Interview.mov"})
    with pytest.raises(ValueError):
        projects.prepare_local(proj["id"], {"keep_cut": True, "aroll": "Missing.mov"})
    projects.prepare_local(proj["id"], {"keep_cut": True, "aroll": "Interview.mov", "graphics": "", "music": "", "library": False})
    assert store.get(proj["job"])["options"]["keep_cut"] is True and store.get(proj["job"])["options"]["aroll"] == "Interview.mov"
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    assert heard == ["Interview.mov"]                                     # the other videos are B-roll even though one talks: never transcribed
    out = next((folder / "Output").glob("*.mp4"))
    assert abs(ffmpeg.duration(out) - 20.0) < 0.25                        # every second of it, the silence and the "um" too
    meta = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    fl = json.loads((folder / ".aivideo" / "files.json").read_text())
    a = next(f for f in fl if f["name"] == "Interview.mov")
    assert meta["intervals"] == [[round(a["start"], 2), round(a["end"], 2)]]
    src_of = lambda c: next(f["name"] for f in fl if f["start"] - 0.01 <= c["src"] < f["end"])      # noqa: E731
    assert meta["cutaways"] and {src_of(c) for c in meta["cutaways"]} <= {"Walk and talk.mov", "Campus.mov"}
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["keep_cut"]["file"] == "Interview.mov" and not plan.get("pending_render") and not plan["hook_text"]
    assert not any(t in fake.calls for t in ("story_edit", "speaker_roles", "video_length", "hook_text"))   # nothing chosen, nothing cut
    # Final Cut: the interview is ONE clip on the main storyline; B-roll only as connected clips above it
    xml = next(folder.glob("*.fcpxml"))
    root = ET.parse(xml).getroot()
    assets = {r.get("id"): r for r in root.iter("asset")}
    def media(el):
        a2 = assets.get(el.get("ref"))
        mr = a2.find("media-rep") if a2 is not None else None
        return (mr.get("src") if mr is not None else a2.get("src", "")) if a2 is not None else ""
    spine = next(root.iter("spine"))
    main = [c for c in spine if c.tag in ("asset-clip", "clip", "mc-clip", "ref-clip")]
    assert len(main) == 1 and media(main[0]).endswith("Interview.mov")
    conn = [c for c in main[0].iter("asset-clip") if c.get("lane") == "1"]
    assert conn and all(not media(c).endswith("Interview.mov") for c in conn)
    from clipfarm.clipper import fcpxml as FX
    problems = FX.validate(xml)
    assert not problems, problems


def test_a_change_to_an_added_broll_video_places_the_broll_again_and_keeps_the_video(tmp_path, monkeypatch):
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = _keep_cut_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    briefs = []

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_plan":
                briefs.append(prompt)
                return _Res({"cutaways": [{"at": 5.0, "scene": 0, "length": 2.0, "why": "x"}]})
            if task == "broll_pick":
                return _Res({"scene": 0, "why": "x"})
            if task in ("revise_request", "clip_edit"):
                raise AssertionError("a kept video is never re-cut")
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    monkeypatch.setattr(LP, "LocalAI", lambda c: AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: [{"text": f"w{i}.", "start": 0.5 + i, "end": 0.9 + i} for i in range(18)])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Open house", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"keep_cut": True, "aroll": "Interview.mov", "graphics": "", "music": "", "library": False})
    runner.make(proj["id"], {})
    runner.revise(proj["id"], "use the campus shot near the end")
    assert store.get(proj["job"])["status"] == "done"
    assert "use the campus shot near the end" in briefs[-1]                # the change steers the B-roll
    outs = sorted((folder / "Output").glob("*.mp4"))
    assert len(outs) == 2 and all(abs(ffmpeg.duration(o) - 20.0) < 0.25 for o in outs)


# ----------------------------------------------------------------------------- a song of your own (music library / dropped on the project)
def _song(path, hz, secs=40):
    import subprocess

    from clipfarm.core import ffmpeg
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={hz}:duration={secs}", "-ac", "2", str(path)],
                   check=True, timeout=120)
    return path


def _band(path, hz):
    import subprocess

    import numpy as np

    from clipfarm.core import ffmpeg
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", str(path), "-af", f"bandpass=f={hz}:width_type=q:w=30,bandpass=f={hz}:width_type=q:w=30",
                        "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True, timeout=120)
    x = np.frombuffer(r.stdout, dtype="<i2").astype("float32") / 32768
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def test_a_song_dropped_on_a_long_video_is_used_and_a_new_song_later_rerenders_the_same_edit(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state"), "music": str(tmp_path / "no-bundled-music")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = _keep_cut_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_plan":
                return _Res({"cutaways": [{"at": 5.0, "scene": 0, "length": 2.0, "why": "x"}]})
            if task == "broll_pick":
                return _Res({"scene": 0, "why": "x"})
            if task in ("music_vibe", "story_edit", "revise_request"):
                raise AssertionError(f"a chosen song needs no AI ({task})")
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    monkeypatch.setattr(LP, "LocalAI", lambda c: AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: [{"text": f"w{i}.", "start": 0.5 + i, "end": 0.9 + i} for i in range(18)])
    runner = LP.LocalRunner(cfg, store, projects)
    first = _song(tmp_path / "Songs" / "First song.wav", 1500)
    second = _song(tmp_path / "Songs" / "Second song.wav", 2500)
    proj = projects.create_local(str(folder), "Open house", "", {"transcriber": "openai"}, {"clips": 0})
    with pytest.raises(ValueError):
        projects.prepare_local(proj["id"], {"song": str(tmp_path / "notes.txt")})
    projects.prepare_local(proj["id"], {"keep_cut": True, "aroll": "Interview.mov", "graphics": "", "music": "", "library": False, "song": str(first)})
    opts = store.get(proj["job"])["options"]
    assert opts["song"] == str(first) and opts["music"] == "auto"                     # a song you chose means music
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done"
    v1 = sorted((folder / "Output").glob("*.mp4"))[-1]
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["music"]["file"] == str(first) and plan["music"]["song"]
    xml = sorted(folder.glob("*.fcpxml"))[-1]
    assert any("First song" in (a.get("name") or "") + (a.get("src") or "") + "".join(m.get("src", "") for m in a.iter("media-rep"))
               for a in ET.parse(xml).getroot().iter("asset"))                        # the song is its own clip in Final Cut
    meta1 = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    runner.music(proj["id"], str(second))                                           # a new song: the same edit rendered again
    assert store.get(proj["job"])["status"] == "done" and store.get(proj["job"])["options"]["song"] == str(second)
    v2 = sorted((folder / "Output").glob("*.mp4"))[-1]
    meta2 = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    assert v2 != v1 and meta2["intervals"] == meta1["intervals"] and meta2["cutaways"] == meta1["cutaways"]   # nothing re-cut, no B-roll moved
    assert abs(ffmpeg.duration(v2) - ffmpeg.duration(v1)) < 0.1
    assert _band(v2, 2500) - _band(v1, 2500) > 6 and _band(v1, 1500) - _band(v2, 1500) > 6     # the second song replaced the first
    runner.music(proj["id"], "")                                                    # no music at all
    v3 = sorted((folder / "Output").glob("*.mp4"))[-1]
    assert _band(v2, 2500) - _band(v3, 2500) > 6
    assert json.loads((folder / ".aivideo" / "plan.json").read_text())["music"] == {"mode": "none"}
    with pytest.raises(ValueError):
        runner.music(proj["id"], str(tmp_path / "missing.wav"))
