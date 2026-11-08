"""Speed: footage is described several files at once (an account AI) or one at a time (this Mac's own model), B-roll that is plainly B-roll is
described while the talking is transcribed, files the catalogue already described are not looked at again, and renders use as many workers
as the Mac can spare."""
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from clipfarm.clipper import loadgov
from clipfarm.clipper import localproj as LP
from clipfarm.clipper.catalog import rid
from clipfarm.core.config import load_config


def _clip(path: Path, secs=3):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=s=320x180:r=25:d={secs}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


class AI:
    """Answers every describe with one scene, slowly, and remembers how many were being answered at the same moment."""
    def __init__(self, local=False):
        self.vision_local, self.calls, self.now, self.most, self.lock = local, 0, 0, 0, threading.Lock()

    def complete_json(self, task, system, prompt, schema, images=None, **kw):
        with self.lock:
            self.calls += 1
            self.now += 1
            self.most = max(self.most, self.now)
        time.sleep(0.25)
        with self.lock:
            self.now -= 1
        return SimpleNamespace(data={"scenes": [{"start_s": 0.0, "end_s": 2.5, "what": "a test pattern", "quality": 7}]})


def _setup(tmp_path, n=4):
    cfg = load_config()
    runner = LP.LocalRunner(cfg, None, None)
    files = []
    for k in range(n):
        p = tmp_path / f"b{k}.mp4"
        _clip(p)
        files.append({"name": p.name, "path": str(p), "start": k * 10.0, "end": k * 10.0 + 3.0, "dur": 3.0, "size": p.stat().st_size, "w": 320, "h": 180,
                      "fps": 25.0, "audio": False, "broll": True})
    return runner, files, LP.State(tmp_path / "proj")


def test_footage_is_described_several_files_at_once_with_an_account_ai(tmp_path):
    runner, files, st = _setup(tmp_path)
    ai = AI(local=False)
    scenes = runner.describe(files, [], ai, st, lambda f, m: None)
    assert ai.calls == 4 and ai.most >= 2 and all(scenes.get(f["name"]) for f in files)


def test_this_macs_own_picture_model_answers_one_file_at_a_time(tmp_path):
    runner, files, st = _setup(tmp_path)
    ai = AI(local=True)
    runner.describe(files, [], ai, st, lambda f, m: None)
    assert ai.calls == 4 and ai.most == 1


def test_plain_broll_is_described_early_and_not_asked_again(tmp_path):
    runner, files, st = _setup(tmp_path, 3)
    talk = tmp_path / "talk.mp4"
    _clip(talk)
    files.append({"name": "talk.mp4", "path": str(talk), "start": 50.0, "end": 53.0, "dur": 3.0, "size": talk.stat().st_size, "w": 320, "h": 180, "fps": 25.0,
                  "audio": True, "broll": False})
    ai = AI()
    t = runner._describe_early(files, ai, st)
    t.join()
    assert ai.calls == 3                                          # the three B-roll files, not the one with sound
    runner.describe(files, [{"text": "w", "start": 50.2, "end": 50.4}], ai, st, lambda f, m: None)
    assert ai.calls == 4                                          # later: only the file whose talking says it is B-roll (almost nobody speaks in it)
    assert runner._describe_early([f for f in files if f["audio"]], ai, st) is None


def test_files_the_catalogue_already_described_are_not_looked_at_again(tmp_path):
    runner, files, st = _setup(tmp_path, 2)
    rec = {"kind": "video", "size": files[0]["size"], "scenes": [{"start": 0.0, "end": 2.8, "what": "red gym", "quality": 8}]}
    runner.catalog = SimpleNamespace(enabled=lambda: True, load=lambda i: rec if i == rid(files[0]["path"]) else None)
    ai = AI()
    scenes = runner.describe(files, [], ai, st, lambda f, m: None)
    assert ai.calls == 1                                          # only the second file went to the vision model
    assert scenes[files[0]["name"]][0]["what"] == "red gym" and scenes[files[1]["name"]]


def test_render_workers_follow_the_mac(monkeypatch):
    away = loadgov.Reading(idle_s=1e9)
    busy = loadgov.Reading(idle_s=2.0, front_ours=True)
    monkeypatch.setattr(loadgov.SystemSampler, "sample", lambda self: away)
    assert 2 <= loadgov.render_workers(2, 4) <= 4
    monkeypatch.setattr(loadgov.SystemSampler, "sample", lambda self: busy)
    assert loadgov.render_workers(2, 4) <= 2                      # while you work: no more than two
    monkeypatch.setattr(loadgov.SystemSampler, "sample", lambda self: (_ for _ in ()).throw(OSError("no top")))
    assert loadgov.render_workers(2, 4) == 2                       # cannot read the Mac: two, as before


def test_a_broll_shot_past_the_end_of_its_clip_is_fitted_not_a_failed_render():
    """BUG (reported 2026-11-07): "the cut would show a picture that doesn't exist (C0020.MP4 at 3.50s for 2.60s (the file is 3.503 s))" stopped
    the whole render. A B-roll shot is shortened to the picture its clip has, or left out; the speaker's picture stays there."""
    files = [{"name": "C0020.MP4", "path": "/x/C0020.MP4", "start": 100.0, "end": 103.503, "dur": 3.503},
             {"name": "C0021.MP4", "path": "/x/C0021.MP4", "start": 110.0, "end": 118.0, "dur": 8.0}]
    cut = [{"at": 5.0, "length": 2.6, "src": 103.5, "scene": 0}, {"at": 20.0, "length": 3.0, "src": 100.0, "scene": 1},
           {"at": 30.0, "length": 4.0, "src": 115.0, "scene": 2}, {"at": 40.0, "length": 2.0, "src": 900.0, "scene": 3},
           {"at": 50.0, "length": 2.5, "src": 9e6, "photo": "/x/p.jpg", "scene": 4}]
    got, said = LP.fit_to_footage(cut, files)
    by = {c["scene"]: c for c in got}
    assert 0 not in by                                        # 0.003 s of picture left: left out
    assert by[1]["length"] == 3.0                             # fits: untouched
    assert by[2]["length"] == 2.98                            # 3 s left in the clip: shortened
    assert 3 not in by and 4 in by                            # no clip there: left out; a photo stays
    assert len(said) == 3
