"""Your music library (clipper/musiclib.py): songs catalogued once (tempo, energy, the best part), searched in words, picked for a video
(long enough, not the same one every time), a short clip starting at the song's best part, and a finished batch of clips getting a new
song without being re-cut (Engine.resong). Audio is generated with ffmpeg; no model (search falls back to names and measured energy)."""

import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipfarm.clipper import music as MU
from clipfarm.clipper import musiclib as ML
from clipfarm.core import ffmpeg


def _wav(path: Path, expr: str, secs: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=22050:d={secs}", "-ac", "1",
                    str(path)], check=True, timeout=120)
    return path


def _cfg(tmp_path, roots):
    data = {"clipper.music_lib": {"roots": [str(r) for r in roots], "dir": str(tmp_path / "lib")}}
    return SimpleNamespace(root=tmp_path, get=lambda k, d=None: data.get(k, d), path=lambda k: tmp_path / k)


@pytest.fixture
def songs(tmp_path):
    root = tmp_path / "Music"
    # a click every half second: 120 BPM, steady energy, 30 s
    click = _wav(root / "Artlist" / "Driving Upbeat Beat - Ana.wav", "if(lt(mod(t,0.5),0.03),0.8*sin(2*PI*1000*t),0)", 30)
    # quiet for 20 s, loud 20-34 s (the drop), quiet again and fading: 45 s
    rise = _wav(root / "Calm Piano Rise.wav", "(if(between(t,20,34),0.9,0.08)*if(gt(t,40),(45-t)/5,1))*sin(2*PI*220*t)", 45)
    (root / ".DS_Store").write_bytes(b"x" * 30_000)
    return root, click, rise


def test_songs_are_catalogued_once_with_tempo_and_their_best_part(tmp_path, songs):
    root, click, rise = songs
    lib = ML.MusicLibrary(_cfg(tmp_path, [root]))
    r = lib.index()
    assert r["new"] == 2 and r["failed"] == 0 and r["tracks"] == 2
    a = lib.get(ML.track_id(click))
    b = lib.get(ML.track_id(rise))
    assert abs(a["bpm"] - 120) < 6                                                   # the click track's tempo
    assert 18 <= b["best"] <= 22 and b["ending"]                                     # the drop at 20 s; it fades out
    assert abs(b["dur"] - 45) < 0.5 and len(b["energy"]) > 50 and max(b["energy"]) <= 1
    assert lib.index()["new"] == 0                                                   # nothing to do the second time
    os.utime(click, (time.time() + 60, time.time() + 60))
    assert lib.index()["new"] == 1                                                   # a changed file is listened to again
    rise.unlink()
    assert lib.index()["forgotten"] == 1 and lib.stats()["tracks"] == 1


def test_search_by_words_and_by_how_the_song_feels(tmp_path, songs):
    root, click, rise = songs
    lib = ML.MusicLibrary(_cfg(tmp_path, [root]))
    lib.index()
    assert lib.search("calm piano")[0]["path"] == str(rise)
    assert lib.search("energetic")[0]["path"] == str(click)                         # no model: measured energy and tempo decide
    hit = lib.search("upbeat")[0]
    assert hit["path"] == str(click) and hit["bpm"] and len(hit["energy"]) <= 120
    assert {h["path"] for h in lib.search("")} == {str(click), str(rise)}


def test_picking_a_song_for_a_video(tmp_path, songs):
    root, click, rise = songs
    cfg = _cfg(tmp_path, [root])
    lib = ML.MusicLibrary(cfg)
    lib.index()
    assert lib.pick("", length_s=40, seed="a").path == rise                          # long enough for a 40 s video
    assert lib.pick("", length_s=40, seed="a", exclude={rise.name}).path == click   # (a shorter one loops)
    picks = {lib.pick("", seed=str(k)).path for k in range(4)}
    assert picks == {click, rise}                                                    # the least used lately: not always the same song
    ML._LIBS.clear()
    t = MU.pick(cfg, mood="calm")
    assert t and t.license == "your music library" and t.credit == ""               # the library comes before the bundled songs


def test_a_short_clip_starts_at_the_songs_best_part(tmp_path, songs):
    root, click, rise = songs
    cfg = _cfg(tmp_path, [root])
    ML._LIBS.clear()
    ML.MusicLibrary(cfg).index()
    start = MU.clip_start(cfg, MU.song(str(rise)), 15.0, "x")
    assert 17.5 <= start <= 22                                                       # just before the drop
    assert MU.clip_start(cfg, MU.song(str(rise)), 200.0, "x") == MU.start_offset(MU.song(str(rise)), 200.0, "x")   # long videos: as before
    elsewhere = _wav(tmp_path / "Desktop" / "Dropped song.wav", "if(between(t,6,12),0.9,0.1)*sin(2*PI*330*t)", 20)
    lib = ML.MusicLibrary(cfg)
    info = lib.info(elsewhere)                                                       # a song from outside the library: listened to now
    assert info and info["extra"] and 4 <= info["best"] <= 8
    assert all(r["path"] != str(elsewhere) for r in lib.records())                  # ...but it doesn't join the library
    assert MU.song(str(tmp_path / "nope.wav")) is None and MU.song(str(elsewhere)).license == "your own"


def _band_db(path: Path, hz: int) -> float:
    """Loudness of one frequency band of a file's sound (dB), to hear whether a song is in it."""
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", str(path), "-af", f"bandpass=f={hz}:width_type=q:w=30,bandpass=f={hz}:width_type=q:w=30", "-ac", "1",
                        "-ar", "8000", "-f", "s16le", "-"], capture_output=True, timeout=120)
    x = np.frombuffer(r.stdout, dtype="<i2").astype("float32") / 32768
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def test_a_finished_batch_of_clips_gets_a_new_song_without_being_recut(tmp_path, monkeypatch):
    from clipfarm.clipper import moments as M
    from clipfarm.clipper import reframe as R
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    cfg.data["paths"]["music"] = str(tmp_path / "no-bundled-music")
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {"music": ""})
    d = store.dir(job["id"])
    src = d / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=40",
                    "-f", "lavfi", "-i", "sine=f=300:d=40", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([[]] * 4, 4))
    words = [{"text": f"w{i}", "start": 10.5 + i * 0.5, "end": 10.8 + i * 0.5} for i in range(30)]
    (d / "transcript.json").write_text(json.dumps({"words": words}))
    eng = Engine(cfg, store)
    m = M.Moment(start=10.0, end=25.0, first=0, last=1, title="The hook", caption="c", score=8)
    first = eng.render_clip(job["id"], 1, m, words, src, layout="auto", style="none", pace="natural", hook_title=False)
    store.update(job["id"], status="done")
    assert first["music"] is None
    song = _wav(tmp_path / "Song.wav", "if(between(t,8,30),0.9,0.2)*sin(2*PI*1200*t)", 40)
    with pytest.raises(ValueError):
        eng.resong(job["id"], str(tmp_path / "missing.wav"))
    assert eng.resong(job["id"], str(song)) == {"ok": True, "clips": 1}
    for _ in range(600):
        if store.get(job["id"])["status"] == "done":
            break
        time.sleep(0.1)
    j = store.get(job["id"])
    c = j["clips"][0]
    assert j["status"] == "done" and j["options"]["song"] == str(song)
    assert c["music"]["path"] == str(song) and c["music"]["start"] >= 6                # from just before the song's best part
    assert c["source_start"] == first["source_start"] and c["source_end"] == first["source_end"] and c["edit"]["intervals"] == first["edit"]["intervals"]
    out = d / c["file"]
    assert abs(ffmpeg.duration(out) - first["duration"]) < 0.2
    with_song = _band_db(out, 1200)
    eng.resong(job["id"], "")                                                        # and out again
    for _ in range(600):
        if store.get(job["id"])["status"] == "done" and store.get(job["id"])["clips"][0].get("music") is None:
            break
        time.sleep(0.1)
    assert store.get(job["id"])["clips"][0]["music"] is None
    assert with_song - _band_db(d / store.get(job["id"])["clips"][0]["file"], 1200) > 6    # the song is in the sound (ducked under the talking)


def test_with_the_music_model_search_listens_to_the_songs(tmp_path, songs):
    root, click, rise = songs

    class FakeClap:                                                                   # "hears" a song by its name; words by their meaning
        def audio(self, path, start, dur=20.0):
            return np.array([1.0, 0.0]) if "Beat" in Path(path).name else np.array([0.0, 1.0])

        def text(self, texts):
            return np.array([[0.2, 0.98] if "relax" in t else [0.98, 0.2] for t in texts])
    lib = ML.MusicLibrary(_cfg(tmp_path, [root]), embedder=FakeClap())
    lib.index()
    assert all(lib.get(ML.track_id(p))["emb"] for p in (click, rise))
    assert lib.search("something to relax to")[0]["path"] == str(rise)             # no name matches: the model decides
    assert lib.search("party")[0]["path"] == str(click)
    assert lib.stats()["searchable"]
