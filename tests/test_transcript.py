"""Extract a transcript: paragraphs, timed text and captions from Parakeet's words; jobs that survive a quit."""
import json
import time
from pathlib import Path

import pytest

from clipfarm.clipper import transcript as T


def W(text, start, end):
    return {"text": text, "start": start, "end": end}


WORDS = [W("Hello", 0.0, 0.4), W("everyone,", 0.45, 0.9), W("welcome.", 1.0, 1.5),            # a long pause, then a new thought
         W("Today", 4.0, 4.3), W("we", 4.35, 4.5), W("talk", 4.55, 4.8), W("about", 4.85, 5.1), W("workshop.", 5.15, 5.9),
         W("It", 6.0, 6.1), W("matters", 6.15, 6.6), W("(a", 6.7, 6.8), W("lot).", 6.85, 7.2)]


def test_paragraphs_break_at_long_pauses_and_join_punctuation_cleanly():
    ps = T.paragraphs(WORDS)
    assert [p["text"] for p in ps] == ["Hello everyone, welcome.", "Today we talk about workshop. It matters (a lot)."]
    assert ps[1]["start"] == 4.0 and T.to_timed(ps).startswith("[00:00] Hello")
    assert T.to_txt(ps) == "Hello everyone, welcome.\n\nToday we talk about workshop. It matters (a lot).\n"
    assert T.clock(3725) == "1:02:05" and T.clock(83) == "01:23"


def test_long_runs_break_at_a_sentence_end():
    words = [W(f"w{i}" + ("." if i % 20 == 19 else ""), i * 0.3, i * 0.3 + 0.25) for i in range(300)]
    ps = T.paragraphs(words)
    assert len(ps) >= 2 and all(p["text"].endswith(".") for p in ps[:-1])


def test_captions_are_short_timed_and_numbered():
    srt = T.to_srt(WORDS)
    blocks = srt.strip().split("\n\n")
    assert blocks[0].startswith("1\n00:00:00,000 --> 00:00:01,500\nHello everyone, welcome.")
    for b in blocks:
        lines = b.split("\n")
        assert len(lines) <= 4 and all(len(x) <= 2 * T.SRT_CHARS for x in lines[2:])
        a, z = lines[1].split(" --> ")
        assert a < z


def _jobs(tmp_path, words=WORDS, fail=None):
    def transcribe(p):
        if fail:
            raise RuntimeError(fail)
        return words

    def download(url, d, progress):
        progress(0.5)
        f = d / "source.m4a"
        f.write_bytes(b"x")
        return f, "A talk from a link"
    out = tmp_path / "Movies"
    return T.Transcripts(tmp_path / "data", None, lambda: out, transcribe=transcribe, download=download), out


def _wait(j, tid):
    for _ in range(200):
        r = j.get(tid)
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError("never finished")


def test_a_video_becomes_three_files_and_the_words_on_the_page(tmp_path, monkeypatch):
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 7.2)
    j, out = _jobs(tmp_path)
    src = tmp_path / "Interview.mov"
    src.write_bytes(b"x")
    r = _wait(j, j.start(str(src)))
    assert r["status"] == "done" and r["words"] == len(WORDS) and r["title"] == "Interview"
    folder = out / "Transcripts"
    assert sorted(p.name for p in folder.iterdir()) == ["Interview (with times).txt", "Interview.srt", "Interview.txt"]
    assert (folder / "Interview.txt").read_text().startswith("Hello everyone")
    assert r["paragraphs"][0]["text"] == "Hello everyone, welcome."
    r2 = _wait(j, j.start(str(src)))                                                   # a second one never overwrites the first
    assert Path(r2["out"]["txt"]).name == "Interview 2.txt"
    assert [x["id"] for x in j.list()] == [r2["id"], r["id"]] and "paragraphs" not in j.list()[0]


def test_a_link_is_downloaded_and_takes_its_title(tmp_path, monkeypatch):
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 7.2)
    j, out = _jobs(tmp_path)
    r = _wait(j, j.start("https://www.youtube.com/watch?v=abc"))
    assert r["status"] == "done" and r["title"] == "A talk from a link" and (out / "Transcripts" / "A talk from a link.srt").exists()


def test_bad_inputs_and_failures_are_plain_words(tmp_path, monkeypatch):
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 7.2)
    j, _ = _jobs(tmp_path, fail="parakeet-mlx isn't installed")
    with pytest.raises(ValueError, match="choose a video"):
        j.start(str(tmp_path / "nope.mov"))
    (tmp_path / "notes.pdf").write_bytes(b"x")
    with pytest.raises(ValueError, match="isn't a video"):
        j.start(str(tmp_path / "notes.pdf"))
    (tmp_path / "a.mp4").write_bytes(b"x")
    r = _wait(j, j.start(str(tmp_path / "a.mp4")))
    assert r["status"] == "failed" and "parakeet" in r["error"]
    j.remove(r["id"])
    assert j.list() == []
    with pytest.raises(ValueError):
        j.get("../../etc")


def test_one_left_running_when_the_app_quit_goes_again(tmp_path, monkeypatch):
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 7.2)
    j, _ = _jobs(tmp_path)
    (tmp_path / "a.mp4").write_bytes(b"x")
    rec = {"id": "t0123456789", "src": str(tmp_path / "a.mp4"), "title": "a", "status": "working", "stage": "Writing down the words",
           "progress": 0.6, "created": 1.0, "error": None, "out": None}
    (tmp_path / "data" / "t0123456789.json").write_text(json.dumps(rec))
    j.resume()
    assert _wait(j, "t0123456789")["status"] == "done"
