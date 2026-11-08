import subprocess

import numpy as np

from clipfarm.clipper import multicam as MC
from clipfarm.core import ffmpeg

SR = 16000


def _voice(seconds, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    t = 0.0
    while t < seconds - 0.5:
        d = rng.uniform(0.15, 0.7)
        i, j = int(t * SR), int(min(seconds, t + d) * SR)
        y[i:j] = rng.standard_normal(j - i).astype(np.float32) * rng.uniform(0.2, 0.6)
        t += d + rng.uniform(0.05, 0.5)
    return y


def _wav(path, y):
    raw = path.with_suffix(".f32")
    raw.write_bytes(y.astype(np.float32).tobytes())
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw), str(path)], check=True)


def _cam(path, voice, start, dur, tone=0.0):
    """A 'camera' that heard the room from `start` for `dur` seconds, through its own cheap microphone."""
    y = voice[int(start * SR):int((start + dur) * SR)]
    y = np.convolve(y, np.ones(7) / 7, mode="same") + np.random.default_rng(int(start * 10) + 5).standard_normal(len(y)).astype(np.float32) * 0.04
    wav = path.with_suffix(".wav")
    _wav(wav, y)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:size=160x120:rate=10:duration={dur}", "-i", str(wav),
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)], check=True, timeout=120)


def test_angle_labels_come_from_folder_and_file_names_then_camera_signature():
    assert MC.angle_label("Interview/Cam A/C0001.MP4") == "A"
    assert MC.angle_label("Shoot/Camera 2/clip.mov") == "2"
    assert MC.angle_label("B-cam/x.mov") == "B"
    assert MC.angle_label("Angle_3_take1.mp4") == "3"
    assert MC.angle_label("raw/mpro012705.MP4") is None
    files = [{"name": "C0001.MP4", "rel": "C0001.MP4", "w": 3840, "h": 2160, "fps": 24.0}, {"name": "C0002.MP4", "rel": "C0002.MP4", "w": 3840, "h": 2160, "fps": 24.0},
             {"name": "IMG_1.MOV", "rel": "IMG_1.MOV", "w": 1920, "h": 1080, "fps": 29.97}, {"name": "x.mp4", "rel": "Cam A/x.mp4", "w": 1, "h": 1, "fps": 30}]
    a = MC.assign_angles(files)
    assert a["C0001.MP4"] == a["C0002.MP4"] != a["IMG_1.MOV"] and a["x.mp4"] == "Cam A"


def test_two_cameras_and_a_recorder_are_lined_up_by_sound_whatever_their_clocks_say(tmp_path):
    voice = _voice(240, 11)
    rec = tmp_path / "recorder.wav"
    _wav(rec, voice)                                                  # the recorder heard all 240 s from the start
    camA, camB, other = tmp_path / "A.mp4", tmp_path / "B.mp4", tmp_path / "other.mp4"
    _cam(camA, voice, 20.0, 150)                                      # camera A started 20 s into the interview
    _cam(camB, voice, 65.5, 120)                                      # camera B started 65.5 s in and finished after A did
    _cam(other, _voice(200, 99), 10.0, 90)                            # something else entirely
    items = [{"name": "recorder.wav", "path": str(rec), "dur": 240.0, "anchor": True},
             {"name": "A.mp4", "path": str(camA), "dur": 150.0}, {"name": "B.mp4", "path": str(camB), "dur": 120.0},
             {"name": "other.mp4", "path": str(other), "dur": 90.0}]
    got = MC.place_files(items)
    assert abs(got["recorder.wav"]["t0"]) < 0.02
    assert abs(got["A.mp4"]["t0"] - 20.0) < 0.05 and abs(got["B.mp4"]["t0"] - 65.5) < 0.05, got
    assert "other.mp4" not in got                                     # not part of this shoot


def test_cameras_sync_to_each_other_without_a_recorder_even_when_one_starts_first(tmp_path):
    voice = _voice(200, 5)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _cam(a, voice, 40.0, 100)                                         # the longest file = the first reference
    _cam(b, voice, 0.0, 70)                                           # B started 40 s BEFORE A and overlaps it for 30 s
    got = MC.place_files([{"name": "a.mp4", "path": str(a), "dur": 100.0}, {"name": "b.mp4", "path": str(b), "dur": 70.0}])
    assert abs(got["b.mp4"]["t0"]) < 0.05 and abs(got["a.mp4"]["t0"] - 40.0) < 0.05, got        # shifted so the earliest file is at 0


def test_detect_group_then_the_director_cut_list_respects_coverage_and_hold_times(tmp_path):
    voice = _voice(240, 21)
    camA, camB = tmp_path / "Cam A" / "C0001.mp4", tmp_path / "Cam B" / "C0001.mp4"
    camA.parent.mkdir()
    camB.parent.mkdir()
    _cam(camA, voice, 0.0, 200)
    _cam(camB, voice, 50.0, 150)                                       # B only covers 50-200
    mk = lambda p, dur, rel: {"name": rel.replace("/", "_"), "rel": rel, "path": str(p), "dur": dur, "audio": True, "w": 160, "h": 120, "fps": 10.0}   # noqa: E731
    files = [mk(camA, 200.0, "Cam A/C0001.mp4"), mk(camB, 150.0, "Cam B/C0001.mp4")]
    g = MC.detect_group(files, [])
    assert g is not None and [a["id"] for a in g["angles"]] == ["Cam A", "Cam B"]
    b = MC.angle_files(g, "Cam B")[0]
    assert abs(b["t0"] - 50.0) < 0.05 and abs(g["dur"] - 200.0) < 0.1
    # the AI wants B from 10 s (where B has no footage), then A for 2 s, then B for 60-90
    segs = [{"start": 10, "end": 40, "angle": "Cam B"}, {"start": 60, "end": 62, "angle": "Cam A"}, {"start": 62, "end": 90, "angle": "Cam B"}]
    cl = MC.cut_list(g, [(5.0, 90.0)], segs)
    assert cl[0]["start"] == 5.0 and cl[-1]["end"] == 90.0
    assert all(s["angle"] == "Cam A" for s in cl if s["end"] <= 50.0)             # B can't show what it never filmed
    assert all(s["end"] - s["start"] >= MC.MIN_HOLD - 1e-6 for s in cl[:-1])       # the 2 s blip was folded into its neighbour
    assert not any(x["angle"] == "Cam A" and 60 <= x["start"] < 62 for x in cl)
    # a shoot whose cameras never overlap is not multicam
    camC = tmp_path / "Cam C" / "far.mp4"
    camC.parent.mkdir()
    _cam(camC, _voice(100, 99), 0.0, 60)
    assert MC.detect_group([mk(camA, 200.0, "Cam A/C0001.mp4"), mk(camC, 60.0, "Cam C/far.mp4")], []) is None


def test_multicam_final_cut_project_is_valid_for_final_cut(tmp_path):
    from fractions import Fraction

    from clipfarm.clipper import fcpxml as FX
    if not FX.DTD_DIR.exists():
        import pytest
        pytest.skip("Final Cut Pro not installed here - DTD check skipped")
    group = {"angles": [{"id": "Cam A", "files": [{"name": "a1", "path": "/Volumes/X/Shoot/Cam A/C0001.mp4", "t0": 0.0, "dur": 200.0, "w": 3840, "h": 2160, "fps": 24.0, "tc": 61234.5}]},
                        {"id": "Cam B", "files": [{"name": "b1", "path": "/Volumes/X/Shoot/Cam B/C0001.mp4", "t0": 50.0, "dur": 150.0, "w": 1920, "h": 1080, "fps": 29.97},
                                                  {"name": "b2", "path": "/Volumes/X/Shoot/Cam B/C0002.mp4", "t0": 205.0, "dur": 30.0, "w": 1920, "h": 1080, "fps": 29.97}]}],
             "recorders": [{"name": "rec", "path": "/Volumes/X/Shoot/Audio/ZOOM0001.wav", "t0": 0.0, "dur": 240.0}], "dur": 240.0}
    items = [{"kind": "mc", "out": 0, "dur": 8.0, "t": 10.0, "video": "Cam A", "audio": "Recorder", "zoom": 1.12},          # a punch-in that hides a jump cut
             {"kind": "mc", "out": 8.0, "dur": 6.5, "t": 60.0, "video": "Cam B", "audio": "Recorder"},
             {"kind": "clip", "out": 14.5, "dur": 4.0, "path": "/Volumes/X/Shoot/Extra/walk.mov", "start": 3.0, "w": 1920, "h": 1080, "fps": 30.0, "audio": True,
              "zoom": 1.12}]
    music = FX.MusicBed("/Volumes/X/Music/song a.mp3", 120.0, "song a", start=2.0, keyframes=[(0, -14), (1, -24), (6, -24), (7, -14)])
    cut = [{"at": 5.0, "len": 2.0, "path": "/Volumes/X/Shoot/Extra/lake.mov", "start": 1.0, "w": 1920, "h": 1080, "fps": 30.0}]
    xml = FX.build_multicam("Interview", 1920, 1080, Fraction(30000, 1001), group, items, captions=[(0.2, 1.4, "hello there"), (9.0, 10.5, "second cut")],
                            music=music, hook=("Watch this", 3.0), cutaways=cut,
                            titles=[{"at": 0.0, "len": 3.2, "text": "Find Your Fit", "kind": "hook"}, {"at": 9.0, "len": 2.8, "text": "Personal Fittings", "kind": "point"},
                                    {"at": 16.0, "len": 2.5, "text": "The Shoe Shack", "kind": "end"}])
    p = tmp_path / "mc.fcpxml"
    p.write_text(xml)
    assert xml.count("<mc-angle ") == 3 and xml.count("<mc-clip ") == 2 and 'srcEnable="audio"' in xml and 'angleID="Recorder"' in xml
    assert xml.count("<adjust-transform") == 2          # both punch-ins are written (inside the angle's source for a multicam clip)
    assert "<sync-clip" not in xml                      # a plain asset-clip per file: Final Cut rejects a sync-clip inside an angle
    assert xml.count('name="Key point"') == 1 and xml.count('name="End card"') == 1 and xml.count('name="Hook title"') >= 1
    assert 'start="61234.5s"' in xml or 'start="122469/2s"' in xml                  # the clip's own timecode origin is declared
    assert FX.validate(p) == []


def test_switches_only_at_allowed_points_hold_at_least_five_seconds_and_follow_the_speaker():
    from clipfarm.clipper import switching as SW
    words, t = [], 0.0
    for s in range(16):
        for k in range(5):
            words.append({"text": f"w{k}" + ("." if k == 4 else ""), "start": round(t, 2), "end": round(t + 0.4, 2), "who": "guest"})
            t += 0.5
        t += 0.5
    g = {"angles": [{"id": "A", "files": [{"name": "a", "path": "/a", "t0": 0.0, "dur": 100.0}]},
                    {"id": "B", "files": [{"name": "b", "path": "/b", "t0": 0.0, "dur": 100.0}]}], "recorders": [], "dur": 100.0}
    pts = SW.switch_points(words, {"A": 0.9, "B": 0.9})
    segs = [{"start": k * 2.3, "end": k * 2.3 + 2.3, "angle": "AB"[k % 2]} for k in range(int(words[-1]["end"] // 2.3))]   # a director that switches every 2.3 s at odd times
    cl = MC.cut_list(g, [(0.0, words[-1]["end"])], segs, 5.0, points=pts, words=words)
    allowed = [p["t"] for p in pts]
    for a, b in zip(cl, cl[1:]):
        assert any(abs(b["start"] - x) < 0.01 for x in allowed) and not SW.inside_a_word(b["start"], words)
    assert all(s["end"] - s["start"] >= 5.0 - 1e-6 for s in cl[:-1])
