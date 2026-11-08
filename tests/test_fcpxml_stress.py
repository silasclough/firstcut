"""Hostile inputs for the Final Cut project writer (WP13): every output must pass Final Cut's DTD (where this Mac has it) and our own pre-flight."""
from __future__ import annotations

import os
import random
import shutil
import xml.etree.ElementTree as ET
from fractions import Fraction

import pytest

from clipfarm.clipper import fcpextras as XT
from clipfarm.clipper import fcpxml as FX

needs_dtd = pytest.mark.skipif(not FX.has_dtd(), reason="no Final Cut / Motion DTD on this machine")
FRS = {"23.976": Fraction(24000, 1001), "25": Fraction(25), "29.97": Fraction(30000, 1001), "30": Fraction(30), "50": Fraction(50), "59.94": Fraction(60000, 1001), "60": Fraction(60)}


def _clip(p, out, dur, start=0.0, w=1920, h=1080, fps=29.97, audio=True, **kw):
    return {"kind": "clip", "out": out, "dur": dur, "path": str(p), "start": start, "w": w, "h": h, "fps": fps, "audio": audio, "file_dur": start + dur + 50, "tc": 0.0, **kw}


def _seq(p, n, dur=5.0, **kw):
    return [_clip(p, i * dur, dur, start=i * dur * 1.1, **kw) for i in range(n)]


def _check(tmp_path, xml, name="x.fcpxml"):
    f = tmp_path / name
    f.write_text(xml)
    ET.fromstring(xml)
    if FX.has_dtd():
        assert FX.validate(f) == []
    assert XT.preflight(xml) == []


@pytest.fixture()
def cam(tmp_path):
    p = tmp_path / "cam.mp4"
    p.write_bytes(b"x")
    return p


@pytest.mark.parametrize("fr", list(FRS))
@pytest.mark.parametrize("wh", [(3840, 2160), (1280, 720), (1080, 1920)])
def test_every_frame_rate_and_size_is_valid_and_on_the_grid(tmp_path, cam, fr, wh):
    f = FRS[fr]
    _check(tmp_path, FX.build_project("P", wh[0], wh[1], f, None, _seq(cam, 3, 4.7, w=wh[0], h=wh[1], fps=float(f))))


def test_a_two_hour_plan_and_a_one_clip_plan(tmp_path, cam):
    _check(tmp_path, FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 720, 10)), "long.fcpxml")
    _check(tmp_path, FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 1, 10)), "one.fcpxml")


def test_ten_thousand_markers_stay_unique_and_in_the_frame_grid(tmp_path, cam):
    rnd = random.Random(1)
    mk = [{"at": i * 0.04 + 0.001, "kind": rnd.choice(["chapter", "todo", "standard"]), "text": f"m{i} \U0001F3AC & <"} for i in range(10000)]
    xml = FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 100, 5), markers=mk)
    _check(tmp_path, xml)
    assert len(ET.fromstring(xml).findall(".//marker")) + len(ET.fromstring(xml).findall(".//chapter-marker")) == 10000


@pytest.mark.parametrize("name", ['Ünï cödé "quoted" & <amp>.mp4', "emoji \U0001F3AC clip's.mp4", "日本語 ファイル.mp4", "a b%20c#d?e.mp4"])
def test_awkward_file_names_titles_and_notes_survive_the_xml(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"x")
    xml = FX.build_project("My \"video\" & <co> \U0001F3AC", 1920, 1080, FRS["29.97"], None, _seq(p, 2, 5),
                           titles=[{"at": 1.0, "len": 2.0, "text": 'Hé "llo" & <b> \U0001F3AC \'x\'', "kind": "lower"}], project_note='Note "q" & <z> \U0001F3AC',
                           markers=[{"at": 1, "kind": "todo", "text": 'Check "x" & <y>', "note": "n&m"}])
    _check(tmp_path, xml)


def test_broll_with_sound_asked_and_not_asked_slow_motion_and_overrun(tmp_path, cam):
    b = tmp_path / "bro.mp4"
    b.write_bytes(b"x")
    cut = lambda **k: {"at": 2.0, "len": 2.0, "path": str(b), "start": 3.0, "w": 3840, "h": 2160, "fps": 59.94, "file_dur": 30.0, **k}   # noqa: E731
    _check(tmp_path, FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 3, 5), cutaways=[cut(), cut(at=8, nat_db=-18.0), cut(at=12, len=1.0, speed=0.25)]))
    _check(tmp_path, FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 2, 5), cutaways=[cut(at=9.5, len=3.0)]), "over.fcpxml")


def test_a_clip_with_no_audio_and_hlg_footage_and_empty_items(tmp_path, cam):
    _check(tmp_path, FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 3, 5, audio=False)))
    _check(tmp_path, FX.build_project("P", 1920, 1080, FRS["29.97"], None, _seq(cam, 2, 5), source_colorspace="9-18-9 (Rec. 2020 HLG)"), "hlg.fcpxml")
    FX.build_project("P", 1920, 1080, FRS["29.97"], None, [])                 # nothing to cut: must not raise


def test_two_files_with_the_same_name_in_different_folders_get_their_own_assets(tmp_path):
    a, b = tmp_path / "a" / "clip.mp4", tmp_path / "b" / "clip.mp4"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"x")
    xml = FX.build_project("P", 1920, 1080, FRS["29.97"], None, [_clip(a, 0, 5), _clip(b, 5, 5)])
    _check(tmp_path, xml)
    assert len(ET.fromstring(xml).findall(".//asset")) == 2


def test_a_missing_media_file_is_reported_by_the_preflight():
    xml = FX.build_project("P", 1920, 1080, FRS["29.97"], None, [_clip("/nonexistent/x.mp4", 0, 5)])
    assert any("missing" in p for p in XT.preflight(xml))
    assert XT.preflight(xml, files_on_disk=False) == []


@pytest.mark.parametrize("seq_fr", ["29.97", "25"])
def test_sources_at_other_frame_rates_get_start_on_their_own_grid(tmp_path, seq_fr):
    items, t = [], 0
    for i, r in enumerate([23.976, 25.0, 29.97, 50.0, 59.94, 60.0]):
        p = tmp_path / f"r{i}.mp4"
        p.write_bytes(b"x")
        items.append(_clip(p, t, 5, start=7.3 + i, fps=r))
        t += 5
    xml = FX.build_project("P", 1920, 1080, FRS[seq_fr], None, items)
    assert XT.preflight(xml) == []


def _read_only_copy(base):
    here = base / "Mein Ordner ü" / "Final Cut" / "Pröject & Co v1"
    (here / "Media").mkdir(parents=True)
    (here / "a.fcpxml").write_text("x")
    (here / "Media" / "m 1.mp4").write_text("y")
    for p in here.rglob("*"):
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(here, 0o555)
    return here


def test_the_bundle_copy_works_over_a_read_only_earlier_copy(tmp_path):
    from clipfarm.clipper import localproj as LP
    here = _read_only_copy(tmp_path)
    staged = tmp_path / "stage" / "Pröject & Co"
    staged.mkdir(parents=True)
    (staged / "a.fcpxml").write_text("new")
    try:
        got = LP.clear_for_copy(here)
        assert got == here and not here.exists()                       # the read-only earlier copy is gone, the same name is free
        shutil.copytree(staged, got, copy_function=shutil.copy2)
        assert (got / "a.fcpxml").read_text() == "new"
    finally:
        for p in [here, *here.rglob("*")]:
            if p.exists():
                os.chmod(p, 0o755)


def test_when_an_earlier_copy_cannot_be_removed_the_project_goes_to_a_fresh_folder(tmp_path, monkeypatch):
    from clipfarm.clipper import localproj as LP
    here = tmp_path / "Final Cut" / "Proj v1"
    (here).mkdir(parents=True)
    (here / "a.fcpxml").write_text("x")
    real = shutil.rmtree
    monkeypatch.setattr(LP.shutil, "rmtree", lambda p, *a, **k: None if str(p) == str(here) else real(p, *a, **k))
    got = LP.clear_for_copy(here)
    assert got.name == "Proj v1 (2)" and got.parent == here.parent and not got.exists()
    assert (here / "a.fcpxml").exists()                                  # (the old one is untouched)


def test_deliver_uses_the_robust_clear_for_copy():
    import inspect

    from clipfarm.clipper import localproj as LP
    src = inspect.getsource(LP.LocalRunner.deliver)
    assert "clear_for_copy(here)" in src and "rmtree(here, ignore_errors=True)" not in src

@needs_dtd
@pytest.mark.parametrize("seq_fr", ["29.97", "25", "60"])
def test_a_three_angle_multicam_project_with_mixed_frame_rates_passes_the_dtd_and_the_preflight(tmp_path, seq_fr):
    def f(name, t0, dur, w, h, fps, tc=0.0):
        p = tmp_path / name
        p.write_bytes(b"x")
        return {"name": name, "path": str(p), "t0": t0, "dur": dur, "w": w, "h": h, "fps": fps, "tc": tc}
    rec = tmp_path / "ZOOM0001.wav"
    rec.write_bytes(b"x")
    group = {"angles": [{"id": "Cam A", "files": [f("a1.mp4", 0.0, 200.0, 3840, 2160, 23.976, 61234.5)]},
                        {"id": "Cam B", "files": [f("b1.mp4", 50.0, 150.0, 1920, 1080, 29.97), f("b2.mp4", 205.0, 30.0, 1920, 1080, 29.97)]},
                        {"id": "Cam C", "files": [f("c1.mp4", 20.0, 200.0, 1920, 1080, 50.0)]}],
             "recorders": [{"name": "rec", "path": str(rec), "t0": 0.0, "dur": 240.0}], "dur": 240.0}
    items = [{"kind": "mc", "out": 0.0, "dur": 8.0, "t": 60.0, "video": "Cam A", "audio": "Recorder", "zoom": 1.12},
             {"kind": "mc", "out": 8.0, "dur": 6.5, "t": 70.3, "video": "Cam B", "audio": "Recorder"},
             {"kind": "mc", "out": 14.5, "dur": 5.0, "t": 95.7, "video": "Cam C", "audio": "Recorder"},
             {"kind": "mc", "out": 19.5, "dur": 4.0, "t": 120.2, "video": "Cam A", "audio": "Recorder"}]
    xml = FX.build_project("Three cams", 1920, 1080, FRS[seq_fr], group, items)
    assert xml.count("<mc-angle ") == 4 and xml.count("<mc-clip ") == 4            # three cameras + the recorder; one mc-clip per piece
    _check(tmp_path, xml, "mc3.fcpxml")


@needs_dtd
@pytest.mark.parametrize("fps", [100.0, 120.0, 240.0])
def test_a_high_frame_rate_main_clip_is_valid_and_not_a_preflight_warning(tmp_path, cam, fps):
    items = [_clip(cam, 0, 5, start=3.3, fps=fps), _clip(cam, 5, 5, start=9.9, fps=fps)]
    xml = FX.build_project("Hfr", 1920, 1080, FRS["29.97"], None, items)
    _check(tmp_path, xml, "hfr.fcpxml")


@needs_dtd
def test_b_roll_without_sound_is_video_only_and_with_sound_plays_at_its_level(tmp_path, cam):
    mute = {"at": 2.0, "len": 2.0, "path": str(cam), "start": 1.0, "w": 1920, "h": 1080, "fps": 29.97, "file_dur": 60.0}
    loud = {"at": 6.0, "len": 2.0, "path": str(cam), "start": 5.0, "w": 1920, "h": 1080, "fps": 29.97, "file_dur": 60.0, "nat_db": -20.0}
    xml = FX.build_project("B", 1920, 1080, FRS["29.97"], None, _seq(cam, 2), cutaways=[mute, loud])
    clips = [c for c in ET.fromstring(xml).iter("asset-clip") if c.get("name") == "B-roll"]
    assert len(clips) == 2
    assert clips[0].get("srcEnable") == "video" and clips[0].find("adjust-volume") is None      # a clean "picture only", not a -96 dB fader
    assert clips[1].get("srcEnable") is None and clips[1].find("adjust-volume").get("amount") == "-20dB"
    _check(tmp_path, xml, "broll.fcpxml")
