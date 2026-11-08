"""Footage record v2 (clipper/footage.py, WP2 of docs/plans/editor-improvement-plan.md): what each file is, from cheap signals first."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from clipfarm.clipper import broll as BR
from clipfarm.clipper import footage as FT
from clipfarm.core import ffmpeg

FF = ffmpeg.bin_path("ffmpeg")


def _vid(path: Path, src: str, secs: float = 4.0, audio: bool = False, fps: int = 10, size: str = "320x180") -> Path:
    sep = ":" if "=" in src else "="
    cmd = [FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"{src}{sep}size={size}:rate={fps}:duration={secs}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-shortest"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def _still(path: Path, secs: float = 4.0) -> Path:
    png = path.with_suffix(".png")
    im = Image.new("RGB", (320, 180), (200, 200, 200))
    d = ImageDraw.Draw(im)
    for k in range(0, 320, 20):
        d.line((k, 0, k + 60, 180), fill=(30, 30, 30), width=3)          # (detail: this one is sharp, only frozen)
    im.save(png)
    subprocess.run([FF, "-y", "-v", "error", "-loop", "1", "-framerate", "10", "-t", str(secs), "-i", str(png), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True, timeout=60)
    return path


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    monkeypatch.setattr(FT, "default_models_dir", lambda: None)


def _facts(path: Path, tmp_path: Path, **kw) -> dict:
    return FT.facts_for(path, None, tmp_path / "facts", **kw)


# ---------------------------------------------------------------------------------------------------------------------------- sidecars
def test_sidecars_and_proxies_attach_to_their_media_and_are_not_footage(tmp_path):
    for n in ("C0001.MP4", "C0001M01.XML", "GX010001.MP4", "GL010001.LRV"):
        (tmp_path / n).write_bytes(b"x")
    r = FT.scan_folder(tmp_path)
    assert sorted(p.name for p in r["media"]) == ["C0001.MP4", "GX010001.MP4"]
    side = {s["name"]: s for s in r["sidecars"]}
    assert set(side) == {"C0001M01.XML", "GL010001.LRV"}
    assert all(s["role"] == "proxy" for s in side.values())
    assert side["C0001M01.XML"]["for"] == "C0001.MP4" and side["GL010001.LRV"]["for"] == "GX010001.MP4"
    assert FT.classify({"name": "GL010001.LRV"})[0] == "proxy"


# ----------------------------------------------------------------------------------------------------------------------------- roles
def _with_faces(monkeypatch):
    monkeypatch.setattr(FT, "_face_detector", lambda models_dir, w, h: (lambda frame: [(60.0, 30.0, 90.0, 110.0, 0.9)]))


def test_a_silent_file_is_broll_and_a_talking_face_is_a_roll(tmp_path, monkeypatch):
    silent = _vid(tmp_path / "shelves.mp4", "testsrc2")
    talk = _vid(tmp_path / "talk.mp4", "testsrc2", audio=True)
    _with_faces(monkeypatch)
    monkeypatch.setattr(FT, "vad", lambda path, dur: [[0.2, dur - 0.2]] if Path(path).name == "talk.mp4" else [])
    monkeypatch.setattr(FT, "active_speaker", lambda video, segments, samples=None: 0.82)
    b = _facts(silent, tmp_path)
    a = _facts(talk, tmp_path)
    assert b["role"] == "broll" and "hardly any speech" in b["role_why"]
    assert a["role"] == "a_roll" and "speaker on camera 0.82" in a["role_why"]
    assert a["faces"]["count_median"] == 1 and a["faces"]["on_camera_speaker_ratio"] == 0.82
    assert a["audio"]["present"] and not b["audio"]["present"]


def test_a_clip_where_only_the_interviewer_talks_off_camera_is_voiceover_broll(tmp_path, monkeypatch):
    clip = _vid(tmp_path / "street.mp4", "testsrc2", audio=True)
    _with_faces(monkeypatch)
    monkeypatch.setattr(FT, "vad", lambda path, dur: [[0.1, dur - 0.1]])
    monkeypatch.setattr(FT, "active_speaker", lambda video, segments, samples=None: 0.04)
    r = _facts(clip, tmp_path)
    assert r["role"] == "voiceover_broll" and "not on camera" in r["role_why"]
    assert FT.classify({**r, "faces": {**r["faces"], "on_camera_speaker_ratio": None}})[0] == "a_roll"      # faces not measured: never called B-roll when someone talks


def test_audio_music_and_pictures_get_their_roles(tmp_path):
    song = tmp_path / "Music" / "calm piano track.mp3"
    song.parent.mkdir()
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(song)], check=True, timeout=60)
    r = _facts(song, tmp_path, rel_parts=("Music", song.name))
    assert r["role"] in ("music", "audio") and not r["has_video"]
    rec = {"name": "mix.wav", "has_video": False, "speech": {"ratio": 0.0}}
    assert FT.classify(rec, ("Music", "mix.wav"))[0] == "music"
    assert FT.classify({**rec, "speech": {"ratio": 0.6}}, ("Music", "mix.wav"))[0] == "audio"
    logo = tmp_path / "logo.png"
    Image.new("RGBA", (200, 200), (255, 0, 0, 255)).save(logo)
    assert _facts(logo, tmp_path)["role"] == "logo"
    shot = tmp_path / "beach.jpg"
    import numpy as np
    Image.fromarray((np.random.default_rng(1).random((480, 640, 3)) * 255).astype("uint8")).save(shot)
    assert _facts(shot, tmp_path)["role"] == "photo"


def test_a_screen_recording_is_not_a_camera_clip():
    rec = {"name": "Screen Recording.mov", "has_video": True, "w": 2880, "h": 1800, "speech": {"ratio": 0.0}, "faces": {"count_median": 0, "measured": True},
           "motion": {"type": "static"}, "tags": {"encoder": "QuickTime Player"}}
    assert FT.classify(rec)[0] == "screen"


def test_shot_size_from_face_ratio_follows_the_table():
    assert [FT.shot_size_from_ratio(r) for r in (0.5, 0.3, 0.2, 0.08)] == ["close", "medium_close", "medium", "wide"]
    assert FT.shot_size_from_ratio(0.0) is None and FT.shot_size_from_ratio(None) is None


# ---------------------------------------------------------------------------------------------------------------------------- quality
def test_blur_exposure_black_and_frozen_flags_on_generated_frames(tmp_path):
    sharp = _vid(tmp_path / "sharp.mp4", "testsrc2")
    flat = _vid(tmp_path / "flat.mp4", "color=c=gray")
    black = _vid(tmp_path / "black.mp4", "color=c=black")
    frozen = _still(tmp_path / "frozen.mp4")
    s, f, b, z = (_facts(p, tmp_path) for p in (sharp, flat, black, frozen))
    assert s["quality"]["blur_floor_ok"] is True and not s["quality"]["black"] and not s["quality"]["frozen"]
    assert f["quality"]["blur_floor_ok"] is False                                     # a flat frame has nothing sharp in it
    assert b["quality"]["black"] and b["quality"]["clip_lo"] > 0.9                    # black: flagged, and nearly every pixel is at the bottom
    assert z["quality"]["frozen"] and z["quality"]["blur_floor_ok"] is True           # a still picture is sharp but frozen
    assert s["usable"] and not b["usable"] and not z["usable"]                           # only the sharp, moving, steady clip has stretches worth cutting from


def test_shots_are_split_at_hard_cuts_and_short_ones_merge():
    shots = FT.shots_from_cuts([5.0, 5.2, 12.0], 20.0)
    assert [(s["start"], s["end"]) for s in shots] == [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)]
    assert [s["cut"] for s in shots] == ["none", "hard", "hard"]


def test_a_cut_in_the_footage_is_found(tmp_path):
    a = _vid(tmp_path / "a.mp4", "testsrc2", 3.0)
    b = _vid(tmp_path / "b.mp4", "color=c=red", 3.0)
    joined = tmp_path / "ab.mp4"
    lst = tmp_path / "l.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n")
    subprocess.run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)], check=True, timeout=60)
    r = _facts(joined, tmp_path)
    assert len(r["shots"]) == 2 and abs(r["shots"][1]["start"] - 3.0) < 0.3
    assert r["shots"][0]["phash"] and r["shots"][1]["phash"]


# ----------------------------------------------------------------------------------------------------------------------------- motion
def test_camera_motion_static_against_pan(tmp_path):
    static = _vid(tmp_path / "static.mp4", "testsrc2", 3.0)                           # (testsrc2 has moving parts, so use a still base)
    still = _still(tmp_path / "still.mp4", 3.0)
    pan = tmp_path / "pan.mp4"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x360:rate=10:duration=3", "-vf", "crop=320:180:x='t*120':y=0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(pan)], check=True, timeout=60)
    ms = FT.motion_fast(still, 3.0, 320, 180)
    mp = FT.motion_fast(pan, 3.0, 320, 180)
    assert ms and ms["type"] == "static" and ms["pan_speed"] < 4
    assert mp and mp["type"] == "pan" and mp["pan_speed"] > 20
    assert static.exists()


def test_slow_motion_eligibility_needs_high_fps_motion_and_no_speech():
    base = {"fps": 59.94, "role": "broll", "speech": {"ratio": 0.0}, "motion": {"energy": 30.0, "shake": 0.1}, "camera": {}}
    assert FT.slowmo_eligible(base, 10.0) is True
    assert FT.slowmo_eligible({**base, "fps": 29.97}, 10.0) is False                  # not enough frames to stretch
    assert FT.slowmo_eligible({**base, "speech": {"ratio": 0.4}}, 10.0) is False      # never slow someone talking
    assert FT.slowmo_eligible({**base, "motion": {"energy": 5.0, "shake": 0.1}}, 10.0) is False     # below the folder's median movement
    assert FT.slowmo_eligible({**base, "motion": {"energy": 30.0, "shake": 0.5}}, 10.0) is False    # shaky
    assert FT.slowmo_eligible({**base, "role": "a_roll"}, 10.0) is False
    assert FT.slowmo_eligible({**base, "camera": {"shutter": 1 / 25}}, 10.0) is False


# ------------------------------------------------------------------------------------------------------------------------------ the cache
def test_facts_cache_is_keyed_by_path_size_and_mtime_not_display_name(tmp_path, monkeypatch):
    clip = _vid(tmp_path / "C0001.MP4", "testsrc2", 2.0)
    calls = []
    real = FT.scan_pass
    monkeypatch.setattr(FT, "scan_pass", lambda p, d: (calls.append(1), real(p, d))[1])
    a = _facts(clip, tmp_path, name="Cam A - C0001.MP4")
    assert len(calls) == 1 and a["name"] == "Cam A - C0001.MP4"
    b = _facts(clip, tmp_path, name="Cam B - C0001.MP4")                             # a different display name for the same file: a cache hit
    assert len(calls) == 1 and b["name"] == "Cam B - C0001.MP4"
    os.utime(clip, (clip.stat().st_atime, clip.stat().st_mtime + 100))              # touched: a different file
    _facts(clip, tmp_path)
    assert len(calls) == 2
    other = tmp_path / "Cam B"
    other.mkdir()
    twin = _vid(other / "C0001.MP4", "testsrc2", 2.5)                                 # two cameras that both wrote C0001.MP4
    assert FT.cache_key(twin) != FT.cache_key(clip)


def test_full_level_only_adds_the_heavy_parts_to_the_fast_record(tmp_path, monkeypatch):
    clip = _vid(tmp_path / "x.mp4", "testsrc2", 2.0)
    fast = _facts(clip, tmp_path)
    assert fast["level"] == "fast"
    calls = []
    monkeypatch.setattr(FT, "scan_pass", lambda p, d: calls.append(1))              # must not run again
    monkeypatch.setattr(FT, "motion_full", lambda p: {"type": "push", "pan_speed": 3.0, "shake": 0.1, "energy": 9.0, "zoom": 0.05})
    full = _facts(clip, tmp_path, level="full", describe=lambda rec: {"model": "fake-vl", "at": 1})
    assert full["level"] == "full" and full["motion"]["type"] == "push" and full["described"]["model"] == "fake-vl" and not calls
    assert _facts(clip, tmp_path)["level"] == "full"                                 # a full record also serves a fast request


def test_the_full_level_waits_for_the_night_window():
    from datetime import datetime
    assert FT.full_allowed(datetime(2026, 11, 7, 22, 0)) and FT.full_allowed(datetime(2026, 11, 7, 3, 0))
    assert not FT.full_allowed(datetime(2026, 11, 7, 14, 0))
    assert FT.describe_cap(datetime(2026, 11, 7, 14, 0)) == 150 and FT.describe_cap(datetime(2026, 11, 7, 23, 0)) == 400
    assert FT.describe_cap(datetime(2026, 11, 7, 14, 0), "full") == 400


# ------------------------------------------------------------------------------------------------------------------------ the whole folder
def test_duplicate_setups_alternate_takes_and_earlier_edits_are_linked():
    h = format(0xF0F0F0F0F0F0F0F0, "016x")
    near = format(0xF0F0F0F0F0F0F0F3, "016x")                                         # 2 bits away
    far = format(0x0123456789ABCDEF, "016x")
    mk = lambda name, ph, dur=10.0: {"name": name, "path": f"/f/{name}", "dur": dur, "shots": [{"start": 0, "end": dur, "phash": [ph, ph, ph]}], "role": "broll",
                                     "speech": {"ratio": 0.0}, "motion": {}, "fps": 30}
    recs = [mk("a.mp4", h), mk("b.mp4", near), mk("c.mp4", far), mk("d.mp4", h, 30.0)]
    FT.finalize(recs)
    assert [r["duplicates_of"] for r in recs] == [None, "a.mp4", None, None]         # b is a's picture again; d is too different in length; c a different shot
    lines = {"t1.mp4": ["I started this company in a garage in two thousand ten.", "We sold the first batch to friends and neighbours."],
             "t2.mp4": ["I started this company in a garage in two thousand ten.", "We sold the first batch to friends and neighbours."],
             "other.mp4": ["The weather turned and the harvest came late this year."]}
    assert FT.alternate_takes(lines) == {"t2.mp4": "t1.mp4"}
    assert FT.repeats_another(lines["t2.mp4"], {"t1.mp4": lines["t1.mp4"]}) and not FT.repeats_another(lines["other.mp4"], {"t1.mp4": lines["t1.mp4"]})
    recs = [{"name": "Final v2.mp4", "role": "a_roll", "speech": {}, "motion": {}, "fps": 30, "dur": 5, "has_video": True},
            {"name": "C0001.MP4", "role": "a_roll", "speech": {}, "motion": {}, "fps": 30, "dur": 5, "has_video": True}]
    FT.finalize(recs, sentences={"Final v2.mp4": lines["t1.mp4"], "C0001.MP4": lines["t2.mp4"]}, edit_named={"Final v2.mp4"})
    assert recs[0]["role"] == "earlier_edit" and recs[1]["role"] != "earlier_edit"


def test_words_settle_speech_and_roles_after_the_transcript(tmp_path):
    f = {"name": "a.mp4", "start": 0.0, "end": 10.0, "dur": 10.0, "path": "/x/a.mp4", "rel": "a.mp4",
         "facts": {"name": "a.mp4", "has_video": True, "w": 1920, "h": 1080, "speech": {"ratio": 0.0}, "faces": {"count_median": 1, "on_camera_speaker_ratio": 0.9, "measured": True},
                   "motion": {}, "tags": {}}}
    words = [{"text": "w", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(16)]
    FT.refine_with_words([f], words)
    assert f["facts"]["speech"]["wps"] == 1.6 and f["facts"]["speech"]["ratio"] == 0.64 and f["facts"]["role"] == "a_roll"


def test_attach_gives_every_scan_entry_its_facts_and_survives_a_bad_file(tmp_path):
    good = _vid(tmp_path / "good.mp4", "testsrc2", 2.0)
    files = [{"name": "good.mp4", "path": str(good), "rel": "good.mp4", "start": 0.0, "end": 2.0, "dur": 2.0},
             {"name": "ghost.mp4", "path": str(tmp_path / "ghost.mp4"), "rel": "ghost.mp4", "start": 2.0, "end": 3.0, "dur": 1.0}]
    FT.attach(files, tmp_path / "facts", workers=2)
    assert files[0]["facts"]["role"] == "broll" and "facts" not in files[1]
    json.dumps(files)                                                                  # (it goes into files.json)
    assert len(json.dumps(files[0]["facts"])) < 6000


# ------------------------------------------------------------------------------------------------------------------------ scene records
def test_describe_returns_the_new_scene_fields_and_old_cache_records_still_load():
    raw = [{"start_s": 0, "end_s": 3.4, "what": "a man pours coffee at a counter", "quality": 7, "shot_size": "Close", "people_on_camera": 1, "is_talking_head": False,
            "on_screen_text": "OPEN", "subject_words": ["Coffee", "counter", "pouring", "coffee"], "mood": "calm", "orientation_ok": "both"},
           {"start_s": 4, "end_s": 8, "what": "a street", "quality": 5, "shot_size": "gigantic", "mood": "grumpy", "is_talking_head": "yes"}]
    sc = BR._valid_scenes(raw, 0.0, 10.0)
    assert sc[0]["shot_size"] == "close" and sc[0]["subject_words"] == ["coffee", "counter", "pouring"] and sc[0]["people_on_camera"] == 1
    assert sc[1]["shot_size"] == "" and sc[1]["mood"] == "neutral" and sc[1]["is_talking_head"] is False        # unusable values become the defaults
    old = {"start": 0.0, "end": 5.0, "what": "a field", "quality": 6.0}                  # a broll.json from before the new fields
    assert BR.with_defaults(old)["is_talking_head"] is False and BR.with_defaults(old)["people_on_camera"] == 0 and BR.with_defaults(old)["what"] == "a field"
    assert "shot_size" in BR.SCHEMA["properties"]["scenes"]["items"]["properties"] and "subject_words" in BR.SYSTEM


def test_the_planner_sees_shot_size_and_camera_movement():
    line = BR.scene_line(3, {"what": "a man pours coffee", "start": 10.0, "end": 13.4, "quality": 7, "shot_size": "close", "motion": "static"})
    assert line == "  S3: a man pours coffee [close, static, 3.4 s, q7]"
    assert BR.scene_line(0, {"what": "x", "start": 0, "end": 2, "quality": 5}) == "  S0: x [2.0 s, q5]"


def test_contact_sheets_are_denser_for_long_files():
    assert BR.sheets_for(30) == 2 and BR.sheets_for(600) == 15 and BR.sheets_for(7200) == 16


def test_long_scenes_split_into_parts_get_their_own_descriptions():
    class Res:
        def __init__(self, data):
            self.data = data

    class AI:
        calls = 0

        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            AI.calls += 1
            assert task == "broll_describe"
            n = int(prompt.split("into ")[1].split(" parts")[0])
            return Res({"scenes": [{"start_s": 100 + k * 10, "end_s": 110 + k * 10, "what": f"part {k} shows something different", "quality": 6} for k in range(n)]})
    scenes = [{"start": 100.0, "end": 180.0, "what": "a long walk", "quality": 6.0}] + \
             [{"start": 200.0 + 30 * k, "end": 220.0 + 30 * k, "what": f"shot {k}", "quality": 6.0} for k in range(5)]
    out = BR.refine_long(AI(), "walk.mp4", scenes, lambda a, b: [(b"jpg", [a, b])])
    assert AI.calls == 3                                                               # six long scenes in the file, but at most 3 extra calls (the longest first)
    assert sum(1 for s in out if s.get("part")) >= 8 and any("part 0 shows" in s["what"] for s in out)
    assert not any("[part" in s["what"] for s in BR.split_long(out) if s.get("part") and "shows something" in s["what"])   # (their own sentences, not the shared one)
    AI.calls = 0
    BR.refine_long(AI(), "walk.mp4", scenes + [{"start": 400.0 + 40 * k, "end": 430.0 + 40 * k, "what": f"long {k}", "quality": 5.0} for k in range(5)],
                   lambda a, b: [(b"jpg", [a, b])])
    assert AI.calls <= BR.LONG_PARTS_MAX_CALLS                                         # never more than three extra calls for one file


def test_scene_extras_carry_the_measured_face_ratio_and_motion():
    rec = {"series": {"t": [0.0, 0.5, 1.0, 1.5, 2.0], "face_h": [0.0, 0.5, 0.5, 0.5, 0.0], "blur": [100] * 5}, "motion": {"type": "pan"},
           "usable": [{"start": 0.0, "end": 2.5, "why": "ok"}]}
    ex = FT.scene_extras(rec, 0.0, 2.5)
    assert ex["motion"] == "pan" and ex["face_ratio"] == 0.5 and ex["shot_size_measured"] == "close" and ex["usable"] is True
    merged = FT.merge_scene({"start": 0.0, "end": 2.5, "what": "a face", "quality": 6.0}, rec)
    assert merged["shot_size"] == "close"                                              # the model gave none: the measured one is used
    assert FT.merge_scene({"start": 0.0, "end": 2.5, "what": "a face", "quality": 6.0, "shot_size": "wide"}, rec)["shot_size"] == "wide"
    assert FT.scene_extras(None, 0, 1) == {}


# ------------------------------------------------------------------------------------------------------------------- scans and the MCP
def test_mts_and_mxf_files_are_scanned(tmp_path):
    from clipfarm.clipper import localproj as LP
    for ext in (".mts", ".mxf", ".avi"):
        assert ext in LP.VIDEO_EXT
    clip = _vid(tmp_path / "C0001.mp4", "testsrc2", 2.0)
    avi = tmp_path / "C0002.avi"
    subprocess.run([FF, "-y", "-v", "error", "-i", str(clip), "-c:v", "mpeg4", str(avi)], check=True, timeout=60)
    assert sorted(f["name"] for f in LP.scan(tmp_path)) == ["C0001.mp4", "C0002.avi"]


def test_each_file_without_a_recorded_time_takes_its_own_date_and_scan_keeps_the_created_time(tmp_path):
    from clipfarm.clipper import localproj as LP
    a = _vid(tmp_path / "a.mp4", "testsrc2", 1.5)
    b = _vid(tmp_path / "b.mp4", "testsrc2", 1.5)
    subprocess.run([FF, "-y", "-v", "error", "-i", str(b), "-c", "copy", "-metadata", "creation_time=2026-01-01T10:00:00Z", str(tmp_path / "b2.mp4")], check=True, timeout=60)
    b.unlink()
    os.utime(a, (1_800_000_000, 1_800_000_000))                                       # a has no recorded time and a late date
    files = LP.scan(tmp_path)
    assert [f["name"] for f in files] == ["b2.mp4", "a.mp4"]                          # b2 (2026-01-01) comes first although mtime order would not say so
    assert files[0]["created"].startswith("2026-01-01") and files[1]["created"] is None


def test_footage_details_shows_the_roles_and_scenes_of_an_edited_folder(tmp_path, monkeypatch):
    from clipfarm.clipper import mcp_server as MS
    clip = _vid(tmp_path / "talk.mp4", "testsrc2", 2.0, audio=True)
    monkeypatch.setattr(MS, "resolve_place", lambda w: Path(w))
    files = [{"name": "talk.mp4", "path": str(clip), "dur": 2.0}]
    assert MS._facts_notes(str(tmp_path), files) == {"talk.mp4": ""}                  # never looked at: nothing invented, nothing written
    assert not (tmp_path / ".aivideo").exists()
    rec = _facts(clip, tmp_path)                                                       # (cache dir elsewhere)
    FT._store(tmp_path / ".aivideo" / "facts", clip, {**rec, "role": "a_roll", "role_why": "speech 0.82"})
    (tmp_path / ".aivideo" / "broll.json").write_text(json.dumps({"talk.mp4": {"size": 1, "scenes": [
        {"start": 0.0, "end": 2.0, "what": "a man talks", "quality": 7, "shot_size": "medium", "is_talking_head": True}]}}))
    out = MS._facts_notes(str(tmp_path), files)["talk.mp4"]
    assert "role a_roll (speech 0.82)" in out and "scenes: 0.0-2.0 s a man talks [medium, talking head]" in out
