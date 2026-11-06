"""The live view (clipper/live.py): events per job for the page while a video is being made, the activity log that updates a repeated
line in place, the words in chunks, small frames next to them, the page's route, and that none of it can break a render."""

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from clipfarm.clipper import live
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs")
    yield s
    live._jobs.clear()


def _job(store):
    return store.create({"app": "studio", "url": "https://example.com/v"}, {}, status="queued")["id"]


def test_events_are_numbered_kept_and_read_after_a_number(store):
    j = _job(store)
    assert live.emit(store, j, "source", title="Talk") == 1
    live.emit(store, j, "moment", title="A", start=1.0, end=20.0, score=8.2)
    r = live.since(store, j, 0)
    assert [e["kind"] for e in r["events"]] == ["source", "moment"] and r["last"] == 2
    assert live.since(store, j, 1)["events"][0]["title"] == "A"
    assert live.since(store, j, 99)["events"][0]["n"] == 1                       # a page from before a reset starts again
    live._save(live._jobs[j], now=True)
    live.forget(j)                                                              # (a restart)
    assert live.since(store, j, 0)["last"] == 2 and live.emit(store, j, "clip", index=1) == 3


def test_the_ring_keeps_the_newest_events(store, monkeypatch):
    monkeypatch.setattr(live, "RING", 5)
    j = _job(store)
    for k in range(9):
        live.emit(store, j, "frame", f=f"live_{k}.jpg")
    evs = live.since(store, j, 0)["events"]
    assert [e["n"] for e in evs] == [5, 6, 7, 8, 9]


def test_a_repeated_line_updates_in_place(store, monkeypatch):
    j = _job(store)
    clock = [1000.0]
    monkeypatch.setattr(live.time, "time", lambda: clock[0])
    assert live.stage(store, j, "Downloading video (3%)", "downloading")
    clock[0] += 0.1
    assert not live.stage(store, j, "Downloading video (4%)")                   # too soon: the next one says it
    clock[0] += 1
    assert live.stage(store, j, "Downloading video (40%)")
    clock[0] += 1
    assert not live.stage(store, j, "Downloading video (40%)")                  # (nothing changed)
    live.stage(store, j, "Transcribing with Parakeet")
    evs = live.since(store, j, 0)["events"]
    assert [e["text"] for e in evs] == ["Downloading video (40%)", "Transcribing with Parakeet"]
    assert evs[0]["replace"] and not evs[1]["replace"] and evs[0]["key"] == live.shape("Downloading video (3%)")


def test_words_arrive_in_chunks_with_a_summary(store, monkeypatch):
    monkeypatch.setattr(live, "MAX_WORDS", 150)
    j = _job(store)
    ws = [{"text": f"w{k}", "start": k * 0.4, "end": k * 0.4 + 0.3} for k in range(200)] + [{"text": "  ", "start": 90}]
    live.words(store, j, ws, of="Interview.mov")
    evs = live.since(store, j, 0)["events"]
    chunks = [e for e in evs if e["kind"] == "words"]
    assert [len(e["words"]) for e in chunks] == [60, 60, 30] and chunks[0]["words"][0] == ["w0", 0.0]
    assert evs[-1]["kind"] == "transcribed" and evs[-1]["count"] == 200 and evs[-1]["of"] == "Interview.mov"
    live.words(store, j, [])                                                    # nothing heard: nothing sent
    assert live.since(store, j, 0)["last"] == len(evs)


def test_frames_are_small_jpegs_and_the_oldest_go(store, monkeypatch):
    monkeypatch.setattr(live, "MAX_FRAMES", 3)
    j = _job(store)
    names = [live.frame(store, j, Image.new("RGB", (1920, 1080), (200, 30, 30))) for _ in range(4)]
    d = store.dir(j)
    assert all(live.NAME.match(n) for n in names)
    assert not (d / names[0]).exists() and all((d / n).exists() for n in names[1:])
    with Image.open(d / names[-1]) as im:
        assert im.size[0] <= 320
    import io
    b = io.BytesIO()
    Image.new("RGB", (800, 600), (0, 90, 200)).save(b, "PNG")
    assert live.frame(store, j, b.getvalue(), width=480)
    assert live.frame(store, j, b"not a picture") == ""                         # never raises
    assert live.frame(store, j, path=d / "missing.mp4", t=3) == ""


def test_a_filmstrip_of_a_video_arrives_frame_by_frame(store):
    j = _job(store)
    src = store.dir(j) / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=green:s=320x180:d=4:r=25",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    live.filmstrip(store, j, src, 4.0, of="the video", n=3)
    for _ in range(100):
        if len([e for e in live.since(store, j, 0)["events"] if e["kind"] == "frame"]) == 3:
            break
        time.sleep(0.05)
    frames = [e for e in live.since(store, j, 0)["events"] if e["kind"] == "frame"]
    assert len(frames) == 3 and all((store.dir(j) / e["f"]).exists() for e in frames)
    assert sorted(e["t"] for e in frames) == [0.7, 2.0, 3.3]


def test_nothing_here_can_break_a_render():
    bad = SimpleNamespace(dir=lambda j: (_ for _ in ()).throw(KeyError(j)))
    assert live.emit(bad, "x", "frame") == 0 and live.stage(bad, "x", "Rendering") == 0
    live.words(bad, "x", [{"text": "hi", "start": 0}])
    assert live.frame(bad, "x", Image.new("RGB", (10, 10))) == ""


def test_the_engine_logs_every_stage(store):
    from clipfarm.clipper.engine import Cancelled, Engine
    j = _job(store)
    fake = SimpleNamespace(store=store)
    Engine._progress(fake, j, "transcribing", "Transcribing with Parakeet", 0.2)
    assert live.since(store, j, 0)["events"][-1]["text"] == "Transcribing with Parakeet"
    store.update(j, cancel=True)
    with pytest.raises(Cancelled):
        Engine._progress(fake, j, "rendering", "Rendering clip 1 of 3", 0.7)


def test_a_long_video_shows_the_lines_it_keeps(store):
    from clipfarm.clipper.localproj import LocalRunner
    j = _job(store)
    r = LocalRunner(SimpleNamespace(get=lambda k, d=None: d), store, None)
    r._live("note", text="before the job is known")                            # (no job yet: nothing, no error)
    r.live_job = j
    sents = [SimpleNamespace(idx=i, text=f"Sentence {i}.", start=i * 4.0, end=i * 4.0 + 3.5) for i in range(6)]
    r._live_lines({"items": [{"kind": "talk", "a": 1, "b": 2}, {"kind": "visual", "start": 0, "end": 3}, {"kind": "talk", "a": 5, "b": 5}]}, sents)
    lines = [e for e in live.since(store, j, 0)["events"] if e["kind"] == "line"]
    assert [x["text"] for x in lines] == ["Sentence 1. Sentence 2.", "Sentence 5."] and lines[0]["start"] == 4.0 and lines[0]["end"] == 11.5


def test_the_page_reads_the_live_view_and_its_frames(store):
    from clipfarm.clipper import studio_web as W
    from clipfarm.clipper.server import MEDIA_OK
    j = _job(store)
    live.emit(store, j, "source", title="Talk")
    live.emit(store, j, "moment", title="A")
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.store = store
    code, body = a.get(["live", j, "1"])
    assert code == 200 and [e["title"] for e in body["events"]] == ["A"] and body["last"] == 2
    assert a.get(["live", j])[1]["last"] == 2
    assert a.get(["live", "nojob"])[0] == 404 and a.get(["live", "../x", "0"])[0] == 404
    assert MEDIA_OK.match("live_12.jpg") and not MEDIA_OK.match("live_x.jpg") and not MEDIA_OK.match("live_1.jpg/..")
