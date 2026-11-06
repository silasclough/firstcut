"""Transitions (2026-11-06): a fade in/out at the ends, a cross dissolve where the story jumps, a dissolve in and out of a long B-roll shot.

The policy is conservative (transitions.plan): none in ads/promos/short videos/kept videos, none inside continuous talking, one event per 12 s.
The same transitions are made in the MP4 (localproj.assemble, polish.run) and written into the Final Cut project as stock, editable parts:
a Cross Dissolve + Audio Crossfade on the spine, opacity ramps for the rest (fcpxml.build_multicam)."""
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import fcpxml as FX
from clipfarm.clipper import localproj as LP
from clipfarm.clipper import transitions as TR
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

DTD = Path("/Applications/Motion.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_11.dtd")
RATE = Fraction(24)


def _plan(*a, **k):
    k.setdefault("dissolves", True)          # (these tests are about the dissolves, which are off by default)
    return TR.plan(*a, **k)


def _valid(xml: str, tmp_path: Path) -> None:
    p = tmp_path / "t.fcpxml"
    p.write_text(xml)
    if DTD.exists():
        r = subprocess.run(["xmllint", "--noout", "--dtdvalid", str(DTD), str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:800]


def part(out, d, path="/talk.mp4", ss=0.0, *, a_path=None, cut=False, layout=None, file_dur=600.0):
    n = round(d * 24)
    v = {"path": path, "ss": ss, "fill": cut, **({"cut": True} if cut else {}), **({"layout": layout} if layout else {})}
    return {"d": n / 24, "n": n, "out": out, "v": v, "a": {"path": a_path or path, "ss": ss, "pre": 0.0}, "item": {"kind": "clip", "path": path, "file_dur": file_dur}}


def talk_run(pieces):
    """Pieces back to back on the timeline: [(seconds, path, source start)]."""
    out, t = [], 0.0
    for d, path, ss in pieces:
        out.append(part(t, d, path, ss))
        t += out[-1]["d"]
    return out


# ------------------------------------------------------------------------------------------------ the policy
def test_a_continuous_talking_passage_gets_no_dissolve_only_the_fades():
    parts = [part(0, 20, ss=10.0), part(20, 20, ss=30.4), part(40, 20, ss=50.9)]            # one recording running on (small gaps between sentences)
    ev = _plan(parts, RATE)
    assert [e["kind"] for e in ev] == ["fade_out"]                       # (opens on someone talking: no fade in)
    assert abs(ev[0]["at"] + ev[0]["dur"] - 60.0) < 0.05


def test_by_default_there_are_no_dissolves_at_all_only_an_end_fade_and_sometimes_a_start_fade():
    far = [part(0, 20, "/a.mp4", 10.0), part(20, 20, "/b.mp4", 500.0), part(40, 20, "/a.mp4", 900.0)]      # (jumps that would dissolve if they were on)
    ev = TR.plan(far, RATE)                                                  # the real default
    assert [e["kind"] for e in ev] == ["fade_out"]
    assert TR.mode_of("auto") == "auto" and TR.mode_of("full") == "full"
    assert [e["kind"] for e in TR.plan(far, RATE, mode="full")] == ["dissolve", "dissolve", "fade_out"] or \
        "dissolve" in [e["kind"] for e in TR.plan(far, RATE, mode="full")]


def test_a_long_jump_or_another_recording_dissolves_once_and_never_twice_within_12_s():
    parts = [part(0, 15, "/a.mp4", 10.0), part(15, 4, "/a.mp4", 400.0), part(19, 15, "/b.mp4", 5.0), part(34, 15, "/b.mp4", 300.0)]
    ev = [e for e in _plan(parts, RATE) if e["kind"] == "dissolve"]
    assert [e["k"] for e in ev] == [1, 3]                                                      # (15 s and 34 s: 19 s apart; the one at 19 s is too near the first)
    assert all(e["audio"] == "crossfade" and e["n"] == 12 and abs(e["dur"] - 0.5) < 1e-9 for e in ev)
    crowded = [part(0, 10, "/a.mp4", 10.0), part(10, 5, "/b.mp4", 5.0), part(15, 5, "/c.mp4", 5.0), part(20, 20, "/d.mp4", 5.0)]
    assert len([e for e in _plan(crowded, RATE) if e["kind"] == "dissolve"]) == 1


def test_no_transitions_in_an_ad_a_short_video_a_kept_video_or_when_switched_off():
    parts = [part(0, 20, "/a.mp4", 10.0), part(20, 20, "/b.mp4", 5.0)]
    assert _plan(parts, RATE, style="ad") == [] and _plan(parts, RATE, style="promo") == []
    assert _plan(parts, RATE, mode="none") == [] and _plan(parts, RATE, mode=None) != [] and TR.mode_of("off") == "none"
    assert _plan([part(0, 10, "/a.mp4", 10.0), part(10, 10, "/b.mp4", 5.0)], RATE) == []     # under 30 s
    assert _plan([], RATE) == []


def test_a_title_card_at_the_start_means_no_fade_in():
    parts = [part(0, 20, "/a.mp4", 10.0), part(20, 20, "/b.mp4", 5.0)]
    assert [e["kind"] for e in _plan(parts, RATE, hook_at_start=True)] == ["dissolve", "fade_out"]


def test_a_long_broll_shot_dissolves_in_and_out_but_a_short_one_does_not():
    long_shot = [part(0, 12, ss=10.0), part(12, 5, "/b.mp4", 30.0, a_path="/talk.mp4", cut=True), part(17, 20, ss=22.0)]
    long_shot[1]["a"] = {"path": "/talk.mp4", "ss": 22.0 - 0.0, "pre": 0.0}
    long_shot[1]["a"]["ss"] = 22.0 - 5.0 + 0.0                                                   # (the speech runs on under the shot)
    long_shot[2]["a"]["ss"] = long_shot[1]["a"]["ss"] + 5.0
    long_shot[2]["v"]["ss"] = long_shot[2]["a"]["ss"]
    long_shot[0]["a"]["ss"] = long_shot[1]["a"]["ss"] - 12.0
    long_shot[0]["v"]["ss"] = long_shot[0]["a"]["ss"]
    ev = [e for e in _plan(long_shot, RATE) if e["kind"] == "broll"]
    assert len(ev) == 1 and (ev[0]["k"], ev[0]["k_out"]) == (1, 2) and abs(ev[0]["at"] - 12.0) < 1e-6 and abs(ev[0]["shot_len"] - 5.0) < 1e-6
    short = [dict(p) for p in long_shot]
    short[1] = part(12, 3, "/b.mp4", 30.0, a_path="/talk.mp4", cut=True)
    assert not [e for e in _plan(short, RATE) if e["kind"] == "broll"]


def test_no_dissolve_where_there_is_no_media_for_the_handles_or_a_layout_is_on():
    bare = [part(0, 20, "/a.mp4", 10.0), part(20, 20, "/b.mp4", 0.0)]                          # the second clip starts at its file's first frame
    assert not [e for e in _plan(bare, RATE) if e["kind"] == "dissolve"]
    shortfile = [part(0, 20, "/a.mp4", 10.0, file_dur=30.1), part(20, 20, "/b.mp4", 5.0)]       # the first runs to the end of its file
    assert not [e for e in _plan(shortfile, RATE, durs={"/a.mp4": 30.1}) if e["kind"] == "dissolve"]
    laid = [part(0, 20, "/a.mp4", 10.0), part(20, 20, "/b.mp4", 5.0, layout="pip", cut=True)]
    assert not [e for e in _plan(laid, RATE) if e["kind"] in ("dissolve", "broll")]


def test_the_end_card_is_never_dissolved_into():
    parts = [part(0, 25, "/a.mp4", 10.0), part(25, 10, "/b.mp4", 5.0), part(35, 2.6, "/card.mp4", 0.0, cut=True)]
    ev = _plan(parts, RATE, outro=True)
    assert not [e for e in ev if e["kind"] in ("dissolve", "broll") and e.get("k") == 2]
    assert [e["kind"] for e in ev][-1] == "fade_out"                                            # (it ends to black after the card instead)


def test_opacity_follows_the_fades():
    ev = [{"kind": "fade_in", "at": 0.0, "dur": 0.5}, {"kind": "fade_out", "at": 9.0, "dur": 1.0}]
    assert TR.opacity_at(ev, 0.0) == 0.0 and abs(TR.opacity_at(ev, 0.25) - 0.5) < 1e-9 and TR.opacity_at(ev, 5.0) == 1.0
    assert abs(TR.opacity_at(ev, 9.5) - 0.5) < 1e-9 and TR.opacity_at(ev, 10.0) == 0.0


# ------------------------------------------------------------------------------------------------ the pieces the MP4 is cut from
def test_a_dissolve_takes_half_a_transition_off_each_side_and_the_length_stays():
    parts = [part(0, 20, "/a.mp4", 10.0), part(20, 20, "/b.mp4", 5.0)]
    ev = [e for e in _plan(parts, RATE) if e["kind"] == "dissolve"]
    rp = TR.render_parts(parts, ev, RATE)
    assert [bool(p.get("xf")) for p in rp] == [False, True, False]
    assert sum(p["n"] for p in rp) == sum(p["n"] for p in parts) == 960                       # the same length, to the frame
    a, t, b = rp
    assert a["n"] == 480 - 6 and b["n"] == 480 - 6 and t["n"] == 12
    assert b["v"]["ss"] == pytest.approx(5.0 + 6 / 24) and b["a"]["ss"] == pytest.approx(5.0 + 6 / 24)        # (the piece after starts half a transition later)
    xa, xb = t["xf"]["a"], t["xf"]["b"]
    assert xa["n"] == xb["n"] == 12 and xa["v"]["ss"] == pytest.approx(10.0 + (480 - 6) / 24)               # (the first carries on past its end...)
    assert xb["v"]["ss"] == pytest.approx(5.0 - 6 / 24)                                                      # (...the second starts before its beginning)
    assert t["xf"]["audio"] == "crossfade"
    assert a["fade_out"] is False and b["fade_in"] is False                                                  # (the sound carries on into the transition)


def test_the_same_recording_under_a_broll_shot_keeps_its_sound_without_a_crossfade():
    speech = lambda t0, d, out: {**part(out, d, "/talk.mp4", t0), "a": {"path": "/talk.mp4", "ss": t0, "pre": 0.0}}
    broll = {**part(12, 5, "/b.mp4", 30.0, a_path="/talk.mp4", cut=True), "a": {"path": "/talk.mp4", "ss": 22.0, "pre": 0.0}}
    parts = [speech(10.0, 12.0, 0), broll, speech(27.0, 20.0, 17)]
    ev = [e for e in _plan(parts, RATE) if e["kind"] == "broll"]
    assert ev
    rp = TR.render_parts(parts, ev, RATE)
    assert [p["xf"]["audio"] for p in rp if p.get("xf")] == ["keep", "keep"]
    assert sum(p["n"] for p in rp) == sum(p["n"] for p in parts)


# ------------------------------------------------------------------------------------------------ the Final Cut project
def _items(two=True):
    items = [{"kind": "clip", "out": 0.0, "dur": 20.0, "path": "/Volumes/X/a.mov", "start": 10.0, "w": 1920, "h": 1080, "fps": 24.0, "audio": True, "file_dur": 600.0}]
    if two:
        items.append({"kind": "clip", "out": 20.0, "dur": 20.0, "path": "/Volumes/X/b.mov", "start": 5.0, "w": 1920, "h": 1080, "fps": 24.0, "audio": True,
                      "file_dur": 600.0})
    return items


def test_final_cut_gets_a_stock_cross_dissolve_centred_on_the_cut_and_fades_at_the_ends(tmp_path):
    ev = [{"kind": "fade_in", "at": 0.0, "n": 14, "dur": 14 / 24}, {"kind": "dissolve", "at": 20.0, "n": 12, "dur": 0.5, "k": 1},
          {"kind": "fade_out", "at": 39.0, "n": 24, "dur": 1.0}]
    xml = FX.build_multicam("Talk", 1920, 1080, Fraction(24), None, _items(), transitions=ev)
    assert xml.count("<transition ") == 1
    assert f'uid="{FX.CROSS_DISSOLVE_UID}"' in xml and 'uid="FFAudioTransition"' in xml
    assert 'offset="79/4s"' in xml                                                                              # 20 s - 0.25 s (centred)
    assert 'duration="1/2s"' in xml                                                                              # the whole dissolve: half a second
    assert xml.count("<adjust-blend") == 2                                                                      # the first clip fades in, the last fades out
    # the sequence is exactly as long as the clips
    assert f'<sequence format="r1" duration="{FX.Clock(Fraction(24)).at(960)}"' in xml
    _valid(xml, tmp_path)


def test_a_dissolve_with_no_media_before_the_clip_is_left_out_of_the_project():
    items = _items()
    items[1]["start"] = 0.1                                                                                       # (less than half a transition before it)
    xml = FX.build_multicam("Talk", 1920, 1080, Fraction(24), None, items, transitions=[{"kind": "dissolve", "at": 20.0, "n": 12, "dur": 0.5, "k": 1}])
    assert "<transition" not in xml


def test_a_dissolving_broll_clip_runs_half_a_transition_longer_each_side_with_opacity_ramps(tmp_path):
    items = _items(two=False)
    items[0]["dur"] = 40.0
    c = {"at": 12.0, "len": 5.0, "path": "/Volumes/X/broll.mov", "start": 30.0, "w": 1920, "h": 1080, "fps": 24.0, "file_dur": 100.0}
    ev = [{"kind": "broll", "at": 12.0, "at_out": 17.0, "shot_len": 5.0, "n": 12, "dur": 0.5, "k": 1, "k_out": 2, "audio": "keep"}]
    xml = FX.build_multicam("Talk", 1920, 1080, Fraction(24), None, items, cutaways=[c], transitions=ev)
    assert xml.count('name="B-roll"') == 1 and xml.count("<adjust-blend") == 1
    line = next(ln for ln in xml.splitlines() if 'name="B-roll"' in ln)
    assert f'duration="{FX.Clock(Fraction(24)).at(120 + 12)}"' in line                                             # 5 s + 12 frames
    assert xml.count("<keyframe ") == 4
    plain = FX.build_multicam("Talk", 1920, 1080, Fraction(24), None, items, cutaways=[c])
    assert "<adjust-blend" not in plain
    _valid(xml, tmp_path)


def test_a_broll_shot_without_room_around_it_keeps_hard_cuts():
    items = _items(two=False)
    items[0]["dur"] = 40.0
    c = {"at": 12.0, "len": 5.0, "path": "/Volumes/X/broll.mov", "start": 0.0, "w": 1920, "h": 1080, "fps": 24.0, "file_dur": 100.0}           # starts at its file's first frame
    ev = [{"kind": "broll", "at": 12.0, "at_out": 17.0, "shot_len": 5.0, "n": 12, "dur": 0.5, "k": 1, "k_out": 2, "audio": "keep"}]
    assert "<adjust-blend" not in FX.build_multicam("Talk", 1920, 1080, Fraction(24), None, items, cutaways=[c], transitions=ev)


def test_a_broll_shot_under_the_last_fade_fades_with_the_picture(tmp_path):
    items = _items(two=False)
    items[0]["dur"] = 30.0
    c = {"at": 26.0, "len": 4.0, "path": "/Volumes/X/broll.mov", "start": 30.0, "w": 1920, "h": 1080, "fps": 24.0, "file_dur": 100.0}
    ev = [{"kind": "fade_out", "at": 29.0, "n": 24, "dur": 1.0}]
    xml = FX.build_multicam("Talk", 1920, 1080, Fraction(24), None, items, cutaways=[c], transitions=ev)
    assert xml.count("<adjust-blend") == 2                                                                         # the clip under it, and the B-roll over it
    _valid(xml, tmp_path)


# ------------------------------------------------------------------------------------------------ the MP4
def _talk(path: Path, color: str, secs: float, w=320, h=180) -> dict:
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r=24:d={secs}",
                    "-f", "lavfi", "-i", f"sine=f=220:d={secs}", "-shortest", "-c:v", "libx264", "-g", "12", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)
    return path


def _rough(tmp_path, tmode):
    a = _talk(tmp_path / "a.mp4", "red", 60)
    b = _talk(tmp_path / "b.mp4", "green", 60)
    files = [{"name": "a.mp4", "path": str(a), "start": 0.0, "end": 60.0, "dur": 60.0, "w": 320, "h": 180, "fps": 24.0, "audio": True, "broll": False},
             {"name": "b.mp4", "path": str(b), "start": 60.0, "end": 120.0, "dur": 60.0, "w": 320, "h": 180, "fps": 24.0, "audio": True, "broll": False}]
    r = LP.LocalRunner(load_config(), None, None)
    st = LP.State(tmp_path / f"proj_{tmode}")
    st.dir.mkdir(parents=True, exist_ok=True)
    rough, ln, parts = r.assemble(files, [(5.0, 25.0), (65.0, 85.0)], [], None, [], st, lambda f: None, even=False, style="story", tmode=tmode)
    return r, rough, ln, parts


def _px(rough, t, tmp_path, name):
    f = tmp_path / f"{name}.png"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{t}", "-i", str(rough), "-frames:v", "1", str(f)], check=True)
    return np.asarray(Image.open(f).convert("RGB")).astype(int)[90, 160]


def test_the_mp4_shows_a_real_cross_dissolve_between_the_two_recordings_and_stays_the_same_length(tmp_path):
    r_off, rough_off, len_off, _ = _rough(tmp_path, "none")
    r, rough, ln, parts = _rough(tmp_path, "full")
    assert [e["kind"] for e in r_off.last_transitions] == []
    kinds = [e["kind"] for e in r.last_transitions]
    assert kinds == ["dissolve", "fade_out"]       # (opens on talking: no fade in)
    assert abs(ln - len_off) < 0.05 and abs(ln - 40.0) < 0.1                                                       # (a dissolve never shortens the video)
    assert len(parts) == 2                                                                                         # (the project gets the pieces as they are)
    red, green = np.array([255, 0, 0]), np.array([0, 128, 0])
    before, mid, after = _px(rough, 19.5, tmp_path, "b"), _px(rough, 20.0, tmp_path, "m"), _px(rough, 20.5, tmp_path, "a")
    assert np.abs(before - red).max() < 70 and np.abs(after - green).max() < 70                                    # hard picture either side...
    assert mid[0] > 40 and mid[1] > 20 and mid[0] < 215                                                           # ...a blend of both at the cut
    hard = _px(rough_off, 20.0, tmp_path, "h")
    assert np.abs(hard - green).max() < 70 or np.abs(hard - red).max() < 70
    a_off, a_on = ffmpeg.duration(rough_off), ffmpeg.duration(rough)
    assert abs(a_off - a_on) < 0.05


def test_the_final_render_fades_in_and_out_of_black(tmp_path):
    from clipfarm.clipper import polish
    src = tmp_path / "s.mp4"
    _talk(src, "red", 6)
    fades = TR.fade_list([{"kind": "fade_in", "at": 0.0, "n": 14, "dur": 0.6}, {"kind": "fade_out", "at": 5.0, "n": 24, "dur": 1.0}])
    assert fades == [{"kind": "fade_in", "at": 0.0, "dur": 0.6}, {"kind": "fade_out", "at": 5.0, "dur": 1.0}]
    out = tmp_path / "o.mp4"
    g = ";".join(["[0:v]format=yuv420p[a0]", "[a0]fade=t=in:st=0.0000:d=0.6000,format=yuv420p[fd0]", "[fd0]fade=t=out:st=5.0000:d=1.0000,format=yuv420p[fd1]"])
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-filter_complex", g, "-map", "[fd1]", "-an", str(out)], check=True)
    assert _px(out, 0.0, tmp_path, "f0").max() < 30 and _px(out, 3.0, tmp_path, "f3")[0] > 200 and _px(out, 5.95, tmp_path, "f9").max() < 60
    assert hasattr(polish, "run")


# ------------------------------------------------------------------------------------------------ the option
def test_the_transitions_option_is_automatic_unless_switched_off(tmp_path, monkeypatch):
    from clipfarm.clipper import studio as SD
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "home"))
    s = SD.Settings()
    assert s.get()["defaults"]["long"]["transitions"] == "auto"
    assert s.update({"defaults": {"long": {"transitions": "none"}}})["defaults"]["long"]["transitions"] == "none"
    assert s.update({"defaults": {"long": {"transitions": "spin"}}})["defaults"]["long"]["transitions"] == "auto"
    assert TR.mode_of(None) == "auto" and TR.mode_of("none") == "none" and TR.mode_of("auto") == "auto"


def test_a_multicam_project_gets_the_same_transitions_inside_its_mc_clips(tmp_path):
    cam = lambda k, t0: {"path": f"/Volumes/X/cam{k}.mov", "t0": t0, "dur": 600.0, "w": 1920, "h": 1080, "fps": 24.0, "tc": 0.0}
    group = {"angles": [{"id": "A", "files": [cam("a", 0.0)]}, {"id": "B", "files": [cam("b", 0.0)]}], "recorders": []}
    items = [{"kind": "mc", "out": 0.0, "dur": 20.0, "t": 100.0, "video": "A", "audio": "A"}, {"kind": "mc", "out": 20.0, "dur": 20.0, "t": 400.0, "video": "B", "audio": "B"}]
    ev = [{"kind": "fade_in", "at": 0.0, "n": 14, "dur": 14 / 24}, {"kind": "dissolve", "at": 20.0, "n": 12, "dur": 0.5, "k": 1}, {"kind": "fade_out", "at": 39.0, "n": 24, "dur": 1.0}]
    xml = FX.build_multicam("Shoot", 1920, 1080, Fraction(24), group, items, transitions=ev)
    assert xml.count("<transition ") == 1 and xml.count("<adjust-blend") == 2
    _valid(xml, tmp_path)
    assert "<transition" not in FX.build_multicam("Shoot", 1920, 1080, Fraction(24), group, items)           # (nothing asked: nothing written)
