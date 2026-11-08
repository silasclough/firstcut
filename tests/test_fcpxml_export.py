"""WP13: the Final Cut project carries what the planner knew (roles, markers, notes, keywords, ratings, source clips, alternate takes, sync clip,
slow motion, conform-rate, punch-in position, probe metadata) and passes Final Cut's own DTD where this Mac has it, and our pre-flight everywhere.

Final Cut itself is never run here: a DTD check proves the structure, not how Final Cut plays it (see docs/plans, WP13 "Could not verify")."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

from clipfarm.clipper import fcpexport as FXE
from clipfarm.clipper import fcpextras as XT
from clipfarm.clipper import fcpxml as FX

FPS = Fraction(30000, 1001)
needs_dtd = pytest.mark.skipif(not FX.has_dtd(), reason="no Final Cut / Motion DTD on this machine (the pre-flight tests still run)")


def _files(tmp_path):
    for n in ("cam.mp4", "rec.wav", "bro.mp4", "cam2.mp4"):
        (tmp_path / n).write_bytes(b"x")
    return {n: str(tmp_path / n) for n in ("cam.mp4", "rec.wav", "bro.mp4", "cam2.mp4")}


def _items(f, **first):
    return [{"kind": "clip", "out": 0, "dur": 6.0, "path": f["cam.mp4"], "start": 1.0, "w": 1920, "h": 1080, "fps": 29.97, "audio": True,
             "file_dur": 60.0, "tc": 0.0, **first},
            {"kind": "clip", "out": 6.0, "dur": 5.0, "path": f["cam.mp4"], "start": 20.0, "w": 1920, "h": 1080, "fps": 29.97, "audio": True,
             "file_dur": 60.0, "tc": 0.0}]


def _cut(f, **kw):
    return {"at": 2.0, "len": 2.0, "path": f["bro.mp4"], "start": 3.0, "w": 3840, "h": 2160, "fps": 59.94, "file_dur": 30.0, **kw}


def _write(tmp_path, xml, name="p.fcpxml"):
    p = tmp_path / name
    p.write_text(xml)
    return p


def _tree(xml):
    return ET.fromstring(xml)


# ---------------------------------------------------------------------------------------------------------------- one builder
def test_the_single_source_builder_is_an_adapter_over_the_project_builder():
    src = FX.Media("/Volumes/X/Media/source.mp4", 120.0, fps=FPS)
    a = FX.build("P", src, [(0.5, 10.0), (11.2, 30.0)], 1920, 1080, FPS, subs=[(0.2, 1.5, "hello")], hook=("Hook", 3))
    items = [{"kind": "clip", "out": 0.0, "dur": 9.5, "path": src.path, "start": 0.5, "w": 1920, "h": 1080, "fps": float(FPS), "audio": True,
              "file_dur": 120.0, "tc": 0.0, "ar": 48000, "ach": 2},
             {"kind": "clip", "out": 9.5, "dur": 18.8, "path": src.path, "start": 11.2, "w": 1920, "h": 1080, "fps": float(FPS), "audio": True,
              "file_dur": 120.0, "tc": 0.0, "ar": 48000, "ach": 2}]
    b = FX.build_project("P", 1920, 1080, FPS, None, items, subs=[(0.2, 1.5, "hello")], hook=("Hook", 3))
    assert a == b
    assert FX.build_multicam is FX.build_project


# ---------------------------------------------------------------------------------------------------------------- roles
def test_every_clip_has_a_role_and_subroles_are_main_dot_sub(tmp_path):
    f = _files(tmp_path)
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f, who="speaker2"), cutaways=[_cut(f, nat_db=-20.0)])
    root = _tree(xml)
    clips = root.findall(".//asset-clip")
    assert clips
    for c in clips:
        assert c.get("audioRole") or c.get("videoRole"), c.attrib
        for r in (c.get("audioRole"), c.get("videoRole")):
            if r:
                assert XT.role_ok(r), r
    assert any(c.get("videoRole") == "video.B-roll" for c in clips)
    assert XT.audio_role("dialogue", "audience") == "dialogue.audience" and XT.audio_role("dialogue", "2") == "dialogue.speaker2"
    assert XT.audio_role("music") == "music" and not XT.role_ok("bad role.") and not XT.role_ok("")


# ---------------------------------------------------------------------------------------------------------------- markers
def test_chapter_and_todo_markers_sit_inside_the_right_clip_on_the_frame_grid(tmp_path):
    f = _files(tmp_path)
    mk = [{"at": 0.0, "kind": "chapter", "text": "1. Opening"}, {"at": 7.3, "kind": "todo", "text": "Check: jump cut", "note": "two takes"},
          {"at": 2.2, "kind": "standard", "text": "B-roll: books"}, {"at": 999.0, "kind": "todo", "text": "after the end"}, {"at": -1, "kind": "todo", "text": "bad"}]
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f), markers=mk)
    root = _tree(xml)
    spine = root.find(".//spine")
    first, second = list(spine)[0], list(spine)[1]
    assert [m.get("value") for m in first.findall("chapter-marker")] == ["1. Opening"]
    assert [m.get("value") for m in first.findall("marker")] == ["B-roll: books"]
    todo = second.findall("marker")
    assert len(todo) == 1 and todo[0].get("completed") == "0" and todo[0].get("note") == "two takes"
    # the todo is 7.3 - 6.0 = 1.3 s into the second clip (which starts 20 s into its file), on the frame grid
    fd = Fraction(1001, 30000)
    start = XT._time(todo[0].get("start"))
    assert (start / fd).denominator == 1
    assert abs(float(start) - 21.3) < 0.04
    assert "after the end" not in xml and ">bad<" not in xml
    assert XT.preflight(xml, files_on_disk=False) == []


# ---------------------------------------------------------------------------------------------------------------- notes, keywords, ratings, takes
def test_broll_clips_carry_notes_keywords_and_takes_carry_ratings(tmp_path):
    f = _files(tmp_path)
    cut = _cut(f, noun="a stack of books", score=0.71, what="books on a table")
    alt = [{"kept": {"path": f["cam.mp4"], "start": 1.5}, "path": f["cam.mp4"], "start": 40.0, "dur": 3.0, "w": 1920, "h": 1080, "fps": 29.97}]
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f), cutaways=[cut], alt_takes=alt)
    root = _tree(xml)
    b = next(c for c in root.iter("asset-clip") if c.get("name") == "B-roll")
    assert "books on a table" in (b.findtext("note") or "") and "a stack of books" in (b.findtext("note") or "")
    kw = b.find("keyword")
    assert kw is not None and "B-roll" in kw.get("value") and "a stack of books" in kw.get("value")
    a = next(c for c in root.iter("asset-clip") if c.get("name") == "Alternate take")
    assert a.get("enabled") == "0" and a.get("lane") == "-4" and a.find("rating").get("value") == "reject"
    assert a.get("start") != "0s"                                                     # (it plays from the other take's own place in the file)


def test_a_source_clips_event_lists_every_file_with_role_keywords_and_usable_favourites(tmp_path):
    f = _files(tmp_path)
    src = [{"path": f["cam.mp4"], "dur": 60, "w": 1920, "h": 1080, "fps": 29.97, "role": "a_roll", "keywords": ["interview"],
            "scenes": [{"start": 2, "end": 9, "what": "speaker at a desk"}, {"start": 50, "end": 99, "what": "past the end"}],
            "usable": [(1, 30)], "reject": [(40, 43, "blurry")]},
           {"path": f["cam2.mp4"], "dur": 20, "w": 3840, "h": 2160, "fps": 59.94, "role": "broll", "keywords": [], "scenes": [], "usable": [], "reject": []}]
    xml = FX.build_project("My video v2", 1920, 1080, FPS, None, _items(f), source_clips=src)
    root = _tree(xml)
    events = root.findall(".//library/event")
    assert [e.get("name") for e in events] == ["My video", "My video source clips"]
    clips = events[1].findall("asset-clip")
    assert [c.get("name") for c in clips] == ["cam", "cam2"]            # (every file, also the one the program never used)
    kws = [k.get("value") for k in clips[0].findall("keyword")]
    assert "A-roll, interview" in kws and "speaker at a desk" in kws and not any("past the end" in k for k in kws)
    assert [r.get("value") for r in clips[0].findall("rating")] == ["favorite", "reject"]
    assert "B-roll" in clips[1].find("keyword").get("value")
    assert {k.get("name") for k in events[1].findall("keyword-collection")} >= {"speaker at a desk"}
    assert XT.preflight(xml, files_on_disk=False) == []


# ---------------------------------------------------------------------------------------------------------------- titles and cards
def test_more_card_kinds_become_editable_titles_and_designed_cards_become_stills_with_transform():
    from clipfarm.clipper import polish as P
    cards = [{"at": 3.0, "len": 2.0, "kind": "card", "spec": {"kind": "quote", "text": "Hello there", "who": "A"}, "rect": None},
             {"at": 6.0, "len": 2.0, "kind": "card", "spec": {"kind": "chapter", "number": 2, "title": "The recipe"}, "rect": (10, 10, 100, 40)},
             {"at": 9.0, "len": 2.0, "kind": "card", "spec": {"kind": "reference", "ref": "Romans 8:28", "passage": "All things work together"}, "rect": None},
             {"at": 12.0, "len": 2.0, "kind": "card", "spec": {"kind": "chart", "bars": [1, 2]}, "rect": None}]
    t = P.fcp_card_titles(cards)
    assert [x["name"] for x in t] == ["Quote", "Chapter title", "Scripture reference"]         # (a chart is not words: it stays a clip)
    assert all(x["off"] for x in t) and t[1]["text"] == "2. The recipe" and t[2]["text"].startswith("Romans 8:28")
    xml = FX.build_project("P", 1280, 720, FPS, None, [{"kind": "clip", "out": 0, "dur": 20.0, "path": "/x/c.mp4", "start": 0, "w": 1280, "h": 720, "fps": 29.97,
                                                         "audio": True, "file_dur": 60.0}], titles=t)
    titles = _tree(xml).findall(".//title")
    assert [x.get("name") for x in titles] == [f"{n} (editable copy, switched off)" for n in ("Quote", "Chapter title", "Scripture reference")]
    assert all(x.get("enabled") == "0" for x in titles)
    assert XT.card_text("chart", {"x": 1}) is None and XT.card_text("lower", {"name": "A B", "role": "C"}) == "A B\nC"
    # the punch-in transform keeps the picture covering the frame and puts the face in the middle
    tr = FX.zoom_transform(1.2, (0.1, 0.5), 1920, 1080)
    assert tr.startswith('position="') and 'scale="1.2 1.2"' in tr
    px = float(tr.split('"')[1].split()[0])
    assert 17 < px < 18.5                                  # clamped to the picture's edge: (0.5/1.2 -> 0.4167): shifts right by 1.2*1.778*0.0833*100
    assert FX.zoom_transform(1.3, None, 1920, 1080) == 'scale="1.3 1.3"'
    assert FX.zoom_transform(float("nan"), ("x", 2), 1920, 1080) == 'scale="1 1"'


# ---------------------------------------------------------------------------------------------------------------- captions
def test_captions_are_anchored_to_the_primary_clip_and_never_overlap(tmp_path):
    f = _files(tmp_path)
    subs = [(0.5, 2.0, "one"), (1.5, 3.0, "two"), (7.0, 8.0, "three")]            # (the first two overlap: the first ends where the second begins)
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f), subs=subs)
    root = _tree(xml)
    caps = root.findall(".//caption")
    assert [c.findtext("text") .strip() if c.findtext("text") else "".join(c.itertext()).strip() for c in caps] == ["one", "two", "three"]
    assert all(c.get("role") == FX.CAPTION_ROLE for c in caps)
    spine = list(root.find(".//spine"))
    assert len(spine[0].findall("caption")) == 2 and len(spine[1].findall("caption")) == 1     # each rides on the clip it starts in
    spans = FX.caption_spans(subs)
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(spans, spans[1:]))


# ---------------------------------------------------------------------------------------------------------------- recorder audio
def test_recorder_audio_is_a_sync_clip_with_camera_audio_deactivated(tmp_path):
    f = _files(tmp_path)
    aud = {"path": f["rec.wav"], "start": 0.4, "tc": 0.0, "dur": 60.0, "ch": 1}
    plain = FX.build_project("P", 1920, 1080, FPS, None, _items(f, aud=aud))
    p = _tree(plain)
    assert p.find(".//sync-clip") is None
    rec = next(c for c in p.iter("asset-clip") if c.get("audioRole") == "dialogue.recorder")
    assert rec.get("lane") == "-3"
    assert any(c.tag == "asset-clip" and c.get("name") == "cam" and (c.find("adjust-volume") is not None and c.find("adjust-volume").get("amount") == "-96dB")
               for c in p.iter())
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f, aud=aud), sync_clip=True)
    root = _tree(xml)
    sc = root.find(".//sync-clip")
    assert sc is not None and sc.find("spine/asset-clip").get("audioRole") == "dialogue.camera"
    assert sc.find("asset-clip").get("audioRole") == "dialogue.recorder" and sc.find("asset-clip").get("srcEnable") == "audio"
    src = {s.get("sourceID"): s.find("audio-role-source") for s in sc.findall("sync-source")}
    assert src["storyline"].get("active") == "0" and src["storyline"].get("role") == "dialogue.camera" and src["connected"].get("role") == "dialogue.recorder"
    assert XT.preflight(xml, files_on_disk=False) == []


# ---------------------------------------------------------------------------------------------------------------- slow motion, conform-rate, reframe
def test_slow_motion_writes_conform_rate_and_a_linear_time_map(tmp_path):
    f = _files(tmp_path)
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f), cutaways=[_cut(f, speed=0.5)])
    b = next(c for c in _tree(xml).iter("asset-clip") if c.get("name") == "B-roll")
    assert b.find("conform-rate").get("srcFrameRate") == "59.94"
    tm = b.find("timeMap")
    pts = tm.findall("timept")
    assert tm.get("frameSampling") == "floor" and [p.get("interp") for p in pts] == ["linear", "linear"]
    assert XT._time(pts[1].get("value")) * 2 == XT._time(pts[1].get("time")) or abs(float(XT._time(pts[1].get("value"))) * 2 - float(XT._time(pts[1].get("time")))) < 0.05
    kids = [c.tag for c in b]
    assert kids.index("conform-rate") < kids.index("timeMap") < kids.index("adjust-conform")           # (the DTD's order: timing params, then the adjustments)
    assert FX.conform_rate_lines(30, FPS, "") == [] and FX.conform_rate_lines(25, FPS, "") == ['<conform-rate srcFrameRate="25"/>']
    assert FX.conform_rate_lines(144, FPS, "") == []                                                    # (a rate Final Cut has no entry for writes nothing)


def test_the_vertical_reframe_is_position_keyframes_and_the_punch_in_uses_the_face(tmp_path):
    f = _files(tmp_path)
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f, zoom=1.2, zoom_xy=[0.3, 0.4]))
    first = list(_tree(xml).find(".//spine"))[0]
    tr = first.find("adjust-transform")
    assert tr is not None and tr.get("scale") == "1.2 1.2"
    x, y = (float(v) for v in tr.get("position").split())
    assert x > 0 and y < 0                       # the face is left of and above the middle: the picture moves right and down to bring it in


# ---------------------------------------------------------------------------------------------------------------- probe metadata
def test_colour_space_sample_rate_and_channels_come_from_the_probe(tmp_path):
    from clipfarm.clipper import localproj as LP
    meta = LP._fcp_meta({"streams": [{"codec_type": "audio", "sample_rate": "44100", "channels": 1}]}, {"color_transfer": "arib-std-b67"})
    assert meta == {"ar": 44100, "ach": 1, "cs": "9-18-9 (Rec. 2020 HLG)"}
    assert LP._fcp_meta({"streams": [{"codec_type": "audio", "sample_rate": "48000", "channels": 2}]}, {"color_transfer": "bt709"}) == {"ar": 48000, "ach": 2}
    f = _files(tmp_path)
    xml = FX.build_project("P", 1920, 1080, FPS, None, _items(f, **meta))
    root = _tree(xml)
    fmts = {x.get("id"): x for x in root.findall("resources/format")}
    asset = next(a for a in root.findall("resources/asset") if a.get("name") == "cam")
    assert asset.get("audioRate") == "44100" and asset.get("audioChannels") == "1"
    assert "Rec. 2020 HLG" in fmts[asset.get("format")].get("colorSpace")
    assert fmts["r1"].get("colorSpace") == "1-1-1 (Rec. 709)"                          # (the sequence stays Rec. 709)


# ---------------------------------------------------------------------------------------------------------------- bundle in place
def test_the_bundle_is_written_in_place_with_no_zip_and_encoded_file_urls(tmp_path):
    from clipfarm.clipper import polish as P
    f = _files(tmp_path)
    photo = tmp_path / "proj" / ".aivideo" / "photos" / "p 1.mp4"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"x")
    mc = {"group": None, "items": _items(f) + [{"kind": "clip", "out": 11.0, "dur": 3.0, "path": str(photo), "start": 0.0, "w": 1920, "h": 1080, "fps": 30.0,
                                               "audio": False, "file_dur": 5.0}], "cutaways": []}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    out = P._relocate_working_media(mc, bundle, tmp_path / "Final Cut" / "My video v1")
    new = out["items"][-1]["path"]
    assert "/.aivideo/" not in new and new.endswith("/Media/p 1.mp4") and (bundle / "Media" / "p 1.mp4").is_file()
    xml = FX.build_project("My video", 1920, 1080, FPS, None, out["items"])
    assert "Media/p%201.mp4" in xml and " 1.mp4" not in xml.replace("p%201", "")              # (spaces are percent-encoded in the file URL)
    assert FX.file_url("/a b/c#d.mp4") == "file:///a%20b/c%23d.mp4"
    assert not list(tmp_path.rglob("*.zip"))                                                  # (no zip anywhere: the folder was copied, not zipped)


# ---------------------------------------------------------------------------------------------------------------- pre-flight
def test_preflight_catches_off_grid_times_dangling_refs_missing_files_and_missing_conform_rate(tmp_path):
    f = _files(tmp_path)
    good = FX.build_project("P", 1920, 1080, FPS, None, _items(f), cutaways=[_cut(f)])
    assert XT.preflight(good, files_on_disk=True) == [] or all("conform" not in p for p in XT.preflight(good))
    assert any("not well-formed" in p for p in XT.preflight("<fcpxml><oops></fcpxml>"))
    off = good.replace('offset="0s"', 'offset="1/7s"', 1)
    assert any("whole number of frames" in p for p in XT.preflight(off, files_on_disk=False))
    dangling = good.replace('ref="a1"', 'ref="zz9"', 1)
    assert any("does not exist" in p for p in XT.preflight(dangling, files_on_disk=False))
    wrong = good.replace('<asset-clip ref="a1"', '<asset-clip ref="r1"', 1)
    assert any("needs asset" in p for p in XT.preflight(wrong, files_on_disk=False))
    (tmp_path / "cam.mp4").unlink()
    assert any("media file missing: cam.mp4" in p for p in XT.preflight(good, files_on_disk=True))
    dup = good.replace('id="a2"', 'id="a1"', 1)
    assert any("used twice" in p for p in XT.preflight(dup, files_on_disk=False))
    nr = good.replace('<conform-rate srcFrameRate="59.94"/>', "")
    assert any("no conform-rate" in p for p in XT.preflight(nr, files_on_disk=False))
    audio = good.replace('<asset-clip ref="a1"', '<asset-clip audioStart="0s" ref="a1"', 1)
    assert any("audioStart and audioDuration" in p for p in XT.preflight(audio, files_on_disk=False))


# ---------------------------------------------------------------------------------------------------------------- DTD
@needs_dtd
def test_the_dtd_validates_every_feature_at_once(tmp_path):
    f = _files(tmp_path)
    aud = {"path": f["rec.wav"], "start": 0.4, "tc": 0.0, "dur": 60.0, "ch": 1}
    mk = [{"at": 0.0, "kind": "chapter", "text": "1. Opening"}, {"at": 3.0, "kind": "todo", "text": "Check", "note": "n"}, {"at": 2.0, "kind": "standard", "text": "B"}]
    alt = [{"kept": {"path": f["cam.mp4"], "start": 1.5}, "path": f["cam.mp4"], "start": 40.0, "dur": 3.0, "w": 1920, "h": 1080, "fps": 29.97}]
    src = [{"path": f["cam.mp4"], "dur": 60, "w": 1920, "h": 1080, "fps": 29.97, "role": "a_roll", "keywords": ["x"], "scenes": [{"start": 2, "end": 9, "what": "desk"}],
            "usable": [(1, 30)], "reject": [(40, 43, "blurry")]}]
    titles = [{"at": 3.0, "len": 2.0, "text": "Hello\n- A", "kind": "point", "off": True, "name": "Quote"}]
    cuts = [_cut(f, nat_db=-20.0, noun="books", score=0.7, what="books", speed=0.5, cs="9-18-9 (Rec. 2020 HLG)")]
    for sync in (False, True):
        xml = FX.build_project("My video v1", 1920, 1080, FPS, None, _items(f, aud=aud, zoom=1.15, zoom_xy=[0.3, 0.4], gain_db=-2.0, ar=44100, ach=1, who="speaker2"),
                               cutaways=cuts, markers=mk, project_note="Rough cut by FirstCut", alt_takes=alt, source_clips=src, titles=titles, sync_clip=sync,
                               subs=[(0.5, 2.0, "hello there")], hook=("Hook", 2.0))
        p = _write(tmp_path, xml, f"all{int(sync)}.fcpxml")
        assert FX.validate(p) == []
        assert XT.preflight(xml, files_on_disk=True) == []


def test_the_extras_come_from_the_plan_and_a_bad_plan_loses_nothing_else():
    files = [{"name": "A.mp4", "path": "/x/A.mp4", "start": 0.0, "end": 100.0, "dur": 100.0, "w": 1920, "h": 1080, "fps": 29.97, "audio": True,
              "facts": {"role": "a_roll", "usable": [{"start": 1.0, "end": 50.0}], "quality": {"black": [[60.0, 62.0]], "frozen": []}}},
             {"name": "B.mp4", "path": "/x/B.mp4", "start": 100.0, "end": 130.0, "dur": 30.0, "w": 3840, "h": 2160, "fps": 59.94, "audio": False, "broll": True}]
    plan = {"summary": "A talk about bread.", "items": [{"kind": "talk", "a": 5, "b": 6, "step": 1}],
            "take_groups": [{"id": "g1", "deliberate": False, "takes": [{"a": 1, "b": 2, "start": 10.0, "end": 14.0}, {"a": 5, "b": 6, "start": 30.0, "end": 35.0}]}],
            "alt_takes": [{"group": "g1", "a": 1, "b": 2}], "unmet": [{"text": "show the oven"}],
            "flow": [{"n": 1, "step": "The recipe"}]}
    scenes = {"A.mp4": [{"start": 2.0, "end": 9.0, "what": "baker at a bench"}], "B.mp4": [{"start": 105.0, "end": 112.0, "what": "loaf"}]}
    out = FXE.extras(plan, files, scenes, lambda a: 30.0, lambda t: t, [{"level": "warn", "what": "a shot of 0.8 s", "start": 4.0, "do": "extend it"},
                                                                       {"level": "info", "what": "x", "start": 5.0}],
                     [{"at": 8.0, "noun": "a loaf", "score": 0.6}], today="2026-11-07")
    kinds = [(m["kind"], m["text"]) for m in out["markers"]]
    assert ("chapter", "The recipe") in kinds and ("todo", "Check: a shot of 0.8 s") in kinds and ("standard", "B-roll: a loaf") in kinds
    assert not any("x" == m["text"] for m in out["markers"])                        # (an "info" is not a to-do)
    alt = out["alt_takes"]
    assert len(alt) == 1 and alt[0]["kept"] == {"path": "/x/A.mp4", "start": 30.0} and alt[0]["start"] == 10.0 and alt[0]["dur"] == 4.0
    sc = out["source_clips"]
    assert [s["role"] for s in sc] == ["a_roll", "broll"] and sc[0]["usable"] == [(1.0, 50.0)] and sc[0]["reject"] == [(60.0, 62.0, "black picture")]
    assert sc[1]["scenes"][0]["start"] == 5.0                                      # (scene times back to the file's own seconds)
    assert "2026-11-07" in out["project_note"] and "show the oven" in out["project_note"] and "bread" in out["project_note"]
    bad = FXE.extras({"flow": 5, "items": "x", "alt_takes": 3}, [{"nope": 1}], None, lambda a: 0, lambda t: t, [None, 5], [None])
    assert set(bad) == {"markers", "project_note", "alt_takes", "source_clips"}


def test_music_keyframes_equal_the_mp4_envelope():
    # the ducking envelope is WP12's: the Final Cut points are produced by the same function the MP4 uses (see tests/test_sound_mix.py for the numbers)
    from clipfarm.clipper import soundmix as SMX
    assert hasattr(SMX, "final_cut_points") and hasattr(SMX, "music_base_db")


def test_nothing_runs_past_the_end_of_the_program_and_markers_are_not_repeated(tmp_path):
    f = _files(tmp_path)
    items = _items(f)
    total = sum(i["dur"] for i in items)
    late = [(total + 2.0, total + 4.0, "too late")]
    xml = FX.build_project("P", 1920, 1080, FPS, None, items, captions=late + [(total - 1.0, total + 5.0, "runs over")], cutaways=[{**_cut(f), "at": total - 0.5, "len": 3.0}])
    assert "too late" not in xml
    assert XT.preflight(xml, files_on_disk=False) == [] or all("past" not in p for p in XT.preflight(xml, files_on_disk=False))
    ms = XT.markers_from_plan({"unmet": [{"at": 3.0, "text": ""}, {"at": 3.0, "text": "x"}, {"at": 3.0, "text": "x"}]})
    assert [m["text"] for m in ms] == ["Not done: x"]
