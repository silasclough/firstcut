"""Tester edge cases, 2026-11-06 (second pass): stock rewrite answers, FCPXML text escaping, music library re-measure, voice cleanup
publishing, names and possessives, chapters. `xfail(strict=True)` marks a bug found by the tester (it flips to a failure when fixed:
then remove the mark). Everything is offline; the AI, the pictures and the program are fakes."""
import json
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import fcpxml as FX
from clipfarm.clipper import glossary as G
from clipfarm.clipper import musiclib as ML
from clipfarm.clipper import notes as NT
from clipfarm.clipper import stockfind as SF
from clipfarm.clipper import studio as SD
from clipfarm.clipper import voiceclean as VC
from clipfarm.core import ffmpeg


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for d in (SF._CACHE, SF._VEC, SF._ALT, SF._PENDING):
        d.clear()
    monkeypatch.setattr(SF, "_REWRITE", None)
    monkeypatch.setattr(SF, "_EMBED", None)
    monkeypatch.setattr(SF, "REWRITE_WAIT", 2.0)


# ---------------------------------------------------------------------------------------------- stock rewrite answers
def _alts(monkeypatch, answer):
    monkeypatch.setattr(SF, "_REWRITE", (lambda q, k: answer()) if callable(answer) else (lambda q, k: answer))
    return SF.searches_for("calm piano", "music")


@pytest.mark.parametrize("answer", [None, [], [""], ["  "], [None], ["calm piano"], ["CALM  piano"]])
def test_an_empty_or_repeated_rewrite_leaves_just_the_words(monkeypatch, answer):
    assert _alts(monkeypatch, answer) == ["calm piano"]


def test_bad_json_inside_a_rewrite_and_a_failing_ai_never_raise(monkeypatch):
    assert _alts(monkeypatch, ['["slow piano", "soft keys", "third", "fourth"]']) == ["calm piano", "slow piano", "soft keys"]   # max 3
    SF._ALT.clear()
    assert _alts(monkeypatch, ["[not json"]) == ["calm piano", "[not json"]                    # kept as text, not a crash
    SF._ALT.clear()
    assert _alts(monkeypatch, lambda: 1 / 0) == ["calm piano"]


def test_a_rewrite_that_is_one_string_is_not_split_into_letters(monkeypatch):
    assert _alts(monkeypatch, "slow piano") == ["calm piano", "slow piano"]


def test_rank_survives_a_picture_model_that_answers_nonsense(monkeypatch):
    class Bad:
        def images(self, ims):
            return __import__("numpy").zeros((1, 4))          # one vector for several pictures

        def text(self, t):
            return __import__("numpy").zeros((1, 4))
    monkeypatch.setattr(SF, "_EMBED", lambda: Bad())
    monkeypatch.setattr(SF, "_thumb_image", lambda u: object())
    items = [{"id": str(i), "thumb": f"u{i}", "kind": "photo", "w": 2000, "h": 2000, "what": f"w{i}", "creator": "c"} for i in range(5)]
    assert sorted(i["id"] for i in SF.rank("q", items)) == ["0", "1", "2", "3", "4"]
    assert SF.rank("q", []) == []


def test_results_without_a_title_are_not_all_duplicates(monkeypatch):
    def prov(q, n, page):
        return [{"source": "pexels", "kind": "video", "id": f"pexels:video:{i}", "what": "", "thumb": "", "preview": "", "w": 1920, "h": 1080,
                 "dur": 10.0, "creator": ""} for i in range(5)]
    monkeypatch.setattr(SF, "PROVIDERS", [("video", prov)])
    assert len(SF.search("x", kinds=("video",), n=10)) == 5


def test_search_with_no_words_no_kinds_or_no_providers_is_empty(monkeypatch):
    monkeypatch.setattr(SF, "PROVIDERS", [])
    assert SF.search("") == [] and SF.search("   ") == [] and SF.search(None) == []
    assert SF.search("x", kinds=()) == [] and SF.search("x") == []


# ---------------------------------------------------------------------------------------------- the music library re-measure
def test_a_library_from_before_version_3_is_listened_to_again(tmp_path):
    assert ML.VERSION == 3 and ML.MODEL == "laion/clap-htsat-unfused"
    root = tmp_path / "Music"
    root.mkdir()
    song = root / "Calm Piano.wav"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=12", str(song)], check=True)
    data = {"clipper.music_lib": {"roots": [str(root)], "dir": str(tmp_path / "lib")}}
    cfg = SimpleNamespace(root=tmp_path, get=lambda k, d=None: data.get(k, d), path=lambda k: tmp_path / k)
    lib = ML.MusicLibrary(cfg)
    assert lib.index()["new"] == 1
    rec = lib.get(ML.track_id(song))
    assert rec["v"] == 3
    lib.save({**rec, "v": 2, "emb": [0.5] * 512})                    # what the old model left behind
    assert lib.index()["new"] == 1                                    # stale: measured again
    assert lib.get(ML.track_id(song))["v"] == 3
    assert lib.index()["new"] == 0


# ---------------------------------------------------------------------------------------------- FCPXML text
ITEMS = [{"kind": "clip", "out": 0.0, "dur": 8.0, "path": "/x/talk.mov", "start": 10.0, "w": 1920, "h": 1080, "fps": 24.0, "audio": True}]


@pytest.mark.parametrize("text", ["it's <b> & \"fine\"".replace('"', ""), "a\nb", "café \U0001F600", "x" * 3000])
def test_odd_caption_text_stays_well_formed_xml(text):
    xml = FX.build_multicam("Talk v3", 1920, 1080, Fraction(24), None, ITEMS, subs=[(0.5, 2.0, text)])
    ET.fromstring(xml.split("\n", 2)[2] if xml.startswith("<?xml") else xml)


def test_a_double_quote_in_a_caption_stays_well_formed_xml():
    xml = FX.build_multicam("Talk v3", 1920, 1080, Fraction(24), None, ITEMS, subs=[(0.5, 2.0, 'He said "hi" now')])
    ET.fromstring(xml.split("\n", 2)[2] if xml.startswith("<?xml") else xml)


def test_a_double_quote_in_the_video_name_stays_well_formed_xml():
    xml = FX.build_multicam('The "big" talk v2', 1920, 1080, Fraction(24), None, ITEMS)
    ET.fromstring(xml.split("\n", 2)[2] if xml.startswith("<?xml") else xml)


def test_event_names_and_layout_geometry_edges():
    assert FX.event_name("") == "FirstCut" and FX.event_name(None) == "FirstCut" and FX.event_name("Talk v12") == "Talk"
    assert FX.event_name("My v2 v3") == "My v2"
    for w, h in ((1080, 1920), (1920, 1080), (1000, 1000)):
        g = FX.layout_geometry("pip", w, h)["inset"]
        x, y = (float(v) for v in g["position"].split())
        assert 0 < y < 50 and 0 < x < 100 * w / h / 2             # inside the frame, top right
    assert FX.layout_geometry("", 1920, 1080) == {"inset": None, "broll": None}
    assert FX.srt(None) == "" and FX.caption_spans(None) == []


def test_an_ad_or_promo_never_gets_a_layout_and_unknown_kinds_do_not_either():
    from clipfarm.clipper import localproj as LP
    for style in ("ad", "Ad", "promo", "advertisement", "explainer"):
        got = LP.choose_layouts([{"at": float(k * 10), "length": 9.0} for k in range(4)], 1920, 1080, style)
        assert all("layout" not in c for c in got), style


# ---------------------------------------------------------------------------------------------- voice cleanup
def _fake_halving_program(home: Path) -> None:
    exe = home / "runtime" / "bin" / "deep-filter"
    ff = ffmpeg.bin_path("ffmpeg")
    exe.write_text('#!/bin/sh\nif [ "$1" = "--help" ]; then exit 0; fi\nfor a in "$@"; do last="$a"; done\nout=""; prev=""\n'
                   'for a in "$@"; do if [ "$prev" = "--output-dir" ]; then out="$a"; fi; prev="$a"; done\n'
                   f'"{ff}" -v error -y -i "$last" -af volume=0.1 "$out/$(basename "$last")"\n')
    exe.chmod(0o755)


def test_voice_cleanup_with_a_missing_file_or_backwards_range_is_an_error_not_a_hang(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "runtime" / "bin").mkdir(parents=True)
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    VC._flags.clear()
    _fake_halving_program(h)
    with pytest.raises(OSError):
        VC.cleaned_range(tmp_path / "nope.wav", 1, 2, tmp_path / "c")
    src = tmp_path / "t.wav"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=4", "-ar", "48000", str(src)], check=True)
    with pytest.raises(Exception):
        VC.cleaned_range(src, 50, 60, tmp_path / "c")                     # past the end of the recording
    assert not list((tmp_path / "c").glob("*.part")) if (tmp_path / "c").exists() else True


def test_a_second_caller_never_gets_a_cleaned_copy_that_is_still_being_judged(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "runtime" / "bin").mkdir(parents=True)
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    VC._flags.clear()
    _fake_halving_program(h)
    src = tmp_path / "t.wav"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=8", "-ar", "48000", str(src)], check=True)
    gate = threading.Event()
    monkeypatch.setattr(VC, "_gain", lambda a, b: (gate.wait(5), 0.0)[1])       # "it didn't help": the original is put back
    t = threading.Thread(target=lambda: VC.cleaned_range(src, 2, 4, tmp_path / "c"))
    t.start()
    try:
        deadline = time.time() + 4
        while time.time() < deadline and not list((tmp_path / "c").glob("*.wav")):
            time.sleep(0.05)
        time.sleep(0.2)
        t0 = time.time()
        got = threading.Thread(target=lambda: VC.cleaned_range(src, 2, 4, tmp_path / "c"))
        got.start()
        got.join(0.5)
        assert got.is_alive(), "the second caller returned before the first one had judged the cleaned copy"
    finally:
        gate.set()
        t.join()
        _ = t0


# ---------------------------------------------------------------------------------------------- names
def test_a_name_keeps_its_possessive():
    words = [{"text": "Eli's", "start": 0.0, "end": 0.4}, {"text": "Northfild's", "start": 0.5, "end": 0.9}]
    assert [w["text"] for w in G.correct(words, ["Eli", "Northfield"])] == ["Eli's", "Northfield's"]


def test_names_edge_cases_never_raise():
    assert G.correct([], ["Northfield"]) == [] and G.correct([{"text": "x", "start": 0, "end": 1}], []) == [{"text": "x", "start": 0, "end": 1}]
    assert G.correct([{"start": 0, "end": 1}, {"text": None, "start": 1, "end": 2}], ["Northfield"])
    assert G.clean(None) == [] and G.clean(5) == [] and G.clean(["", "  ", "123", "a\nb"]) == ["a", "b"]
    assert len(G.clean([f"Name{chr(97 + i % 26)}{i}" for i in range(300)])) == G.MAX_TERMS
    assert [w["text"] for w in G.correct([{"text": "Northfeld,", "start": 0, "end": 1}], ["Northfield"])] == ["Northfield,"]


# ---------------------------------------------------------------------------------------------- chapters
def _words(n=120):
    out, t = [], 0.0
    for i in range(n):
        for w in f"Sentence number {i} is here.".split():
            out.append({"text": w, "start": t, "end": t + 0.3})
            t += 0.35
    return out


class _AI:
    def __init__(self, data=None, exc=None):
        self.data, self.exc = data, exc

    def complete_json(self, *a, **k):
        if self.exc:
            raise self.exc
        return SimpleNamespace(data=self.data)


@pytest.mark.parametrize("data", [{}, {"chapters": "abc"}, ["x"], {"summary": "s", "chapters": [{"at": None, "title": "A"}]},
                                  {"summary": "s", "chapters": [{"at": 0, "title": "A"}, {"at": 90, "title": "B"}]},      # only two
                                  {"summary": "s", "chapters": [{"at": 0, "title": "A"}, {"at": 5, "title": "B"}, {"at": 8, "title": "C"}]}])
def test_chapters_the_ai_gets_wrong_fall_back_to_the_transcript(data):
    chs, _ = NT.chapters(_AI(data), _words(), 210.0)
    assert NT.youtube_ok(chs, 210.0)


def test_chapters_with_no_ai_a_failing_ai_and_short_videos():
    assert NT.youtube_ok(NT.chapters(None, _words(), 210.0)[0], 210.0)
    assert NT.chapters(_AI(exc=TimeoutError("x")), _words(), 210.0)[0]
    assert NT.chapters(None, _words(), 20.0) == ([], "") and NT.chapters(_AI({"summary": "s", "chapters": []}), [], 100.0) == ([], "")
    chs, summary = NT.chapters(_AI({"summary": "line1\n\nline2", "chapters": [{"at": "0", "title": "A\nB"}, {"at": "60", "title": "B"}, {"at": "120", "title": "C"}]}),
                               _words(), 210.0)
    assert summary == "line1 line2" and chs[0] == (0.0, "A B")
    assert NT.chapters_txt([(0, "A"), (3700, "B")]) == "00:00 A\n1:01:40 B\n" and NT.chapters_txt([]) == ""


def test_the_quality_report_with_odd_issues():
    md = NT.quality_md([{"level": "bad", "what": "x", "start": 3700, "end": 3705, "do": "d"}, {"level": "zzz", "what": "", "start": None},
                        {"level": "warn", "what": "w", "start": 5, "end": None}], {"duration": 9}, "T {x}")
    assert "1 to fix, 1 to look at" in md and "1:01:40" in md and "loudness ? LUFS" in md
    assert NT.quality_md([], {}).startswith("# Quality check") and "Nothing to fix" in NT.quality_md([], {})
    NT.begin()
    NT.note("warn", "a")
    assert len(NT.take()) == 1 and NT.take() == []


# ---------------------------------------------------------------------------------------------- keys (Settings > Online sources)
def test_a_pasted_key_is_checked_trimmed_and_never_runs_security_for_an_unknown_name(monkeypatch):
    calls = []
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.setattr(SD.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(ValueError):
        SD.key_set("../evil", "abcdefgh12345")
    for bad in ("short", "has space in it 123", "a" * 201, "key;rm -rf", "line\nbreak12345"):
        with pytest.raises(ValueError):
            SD.key_set("pixabay", bad)
    assert calls == []
    SD.key_set("pixabay", "  12345678-abcdef  ")
    assert calls and calls[0][-1] == "12345678-abcdef"
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
