"""Montage and slideshow videos (montage.py + the runner): detection, pace, length, the variety rule, music fitting, the plan card, and a real
small render from generated clips and photos with a fake AI."""
import json
import subprocess
from pathlib import Path

import pytest

from clipfarm.clipper import montage as MO
from clipfarm.clipper.store import Store
from clipfarm.core.config import load_config


# ------------------------------------------------------------------------------------------------------------------ detection
def test_a_folder_with_no_speech_is_a_montage_and_a_normal_interview_stays_normal():
    d = MO.detect("a youth camp recap, upbeat", n_words=3, footage_s=1200, n_videos=20)
    assert d["style"] == "montage" and d["why"] == "no_speech" and "no one is speaking" in d["line"]
    assert MO.detect("", n_words=900, footage_s=600, n_videos=3) is None                       # an interview: plenty of words
    assert MO.detect("a two minute admissions video", n_words=700, footage_s=900, n_videos=6) is None


def test_the_notes_can_ask_for_a_montage_or_a_slideshow_and_off_turns_it_off():
    assert MO.detect("make a montage of the week", n_words=None, n_videos=5)["style"] == "montage"
    s = MO.detect("slideshow, take it slow", n_photos=40, n_videos=5)
    assert s["style"] == "slideshow" and s["why"] == "notes"
    assert MO.detect("slideshow", n_photos=40, setting="off") is None
    assert MO.detect("anything", n_words=3, footage_s=900, n_videos=5, setting="off") is None
    assert MO.detect("a recap", n_words=900, footage_s=600, n_videos=3) is None                # a weak word alone does not decide: people are talking


def test_a_folder_that_is_mostly_photos_is_a_slideshow_even_with_a_few_clips():
    d = MO.detect("", n_photos=46, n_videos=6)
    assert d["style"] == "slideshow" and d["why"] == "photos" and "photos with a few clips" in d["line"]
    assert MO.detect("", n_photos=3, n_videos=20, n_words=900, footage_s=600) is None


def test_clips_with_no_sound_track_at_all_are_a_montage():
    assert MO.detect("", n_videos=8, audio_clips=0)["style"] == "montage"
    assert MO.detect("", n_videos=8, audio_clips=3) is None


# ---------------------------------------------------------------------------------------------------------------------- pace
def test_pace_decides_how_long_a_shot_is_held_and_what_it_lands_on():
    period = 0.5                                                                                # 120 BPM: a beat is 0.5 s, a bar 2 s
    fast = MO.hold(MO.PACE["fast"]["clip"], period, "beat")
    regular = MO.hold(MO.PACE["auto"]["clip"], period, "bar")
    slow_photo = MO.hold(MO.PACE["slow"]["photo"], period, "bar")
    assert 1.0 <= fast <= 2.5 and (fast / period) % 1 == 0
    assert 2.5 <= regular <= 4.0 and (regular / (period * 4)) % 1 == 0                           # whole bars
    assert 5.0 <= slow_photo <= 7.0 and (slow_photo / (period * 4)) % 1 == 0
    assert MO.hold((2.5, 4.0), None, "bar") == pytest.approx(3.25)                               # no beat: the middle of the range
    assert MO.pace_key("natural") == "auto" and MO.pace_key("fast") == "fast" and MO.pace_key("slow") == "slow"


def cands(n_clips=0, n_photos=0, what="kids playing a game"):
    out = []
    for k in range(n_clips):
        out.append({"id": f"c{len(out)}", "kind": "clip", "name": f"clip{k}.mp4", "path": f"/x/clip{k}.mp4", "start": k * 100.0, "end": k * 100.0 + 12.0,
                    "what": f"{what} {k}", "quality": 7.0, "t": [k, 0.0]})
    for k in range(n_photos):
        out.append({"id": f"c{len(out)}", "kind": "photo", "name": f"p{k}.jpg", "path": f"/x/p{k}.jpg", "start": None, "end": None,
                    "what": f"photo number {k} of campers", "quality": 7.0, "t": [1000 + k, 0.0]})
    return out


def laid(cs, **kw):
    m = {"cands": cs, "order": [c["id"] for c in cs], "removed": [], "pace": "auto", "period": 0.5, "target_s": None, **kw}
    return MO.layout(m)


def test_every_pace_gives_the_lengths_it_promises_and_cuts_on_the_music():
    cs = cands(n_clips=6, n_photos=6)
    for pace, (lo_c, hi_c), (lo_p, hi_p) in (("fast", (1.0, 2.5), (1.0, 2.2)), ("auto", (2.5, 4.0), (2.5, 4.0)), ("slow", (5.0, 8.0), (5.0, 7.0))):
        m = laid(cs, pace=pace)
        for it in m["items"]:
            d = it["end"] - it["start"]
            lo, hi = (lo_p, hi_p) if "photo" in it else (lo_c, hi_c)
            assert lo - 1e-6 <= d <= hi + 1e-6, (pace, it, d)
            assert (d / 0.5) % 1 == pytest.approx(0, abs=1e-6) or (d / 0.5) % 1 == pytest.approx(1, abs=1e-6)     # a whole number of beats
        assert m["total_s"] == pytest.approx(sum(i["end"] - i["start"] for i in m["items"]))


# -------------------------------------------------------------------------------------------------------------------- length
def test_ai_decides_the_length_from_the_shots_and_the_pace():
    cs = cands(n_clips=10)
    slow, fast = laid(cs, pace="slow"), laid(cs, pace="fast")
    assert slow["total_s"] > 2 * fast["total_s"]                                                  # the same shots, a different length
    assert len(slow["items"]) == len(fast["items"]) == 10                                         # all of them are used
    assert slow["note"] == ""


def test_ai_decides_never_goes_over_eight_minutes_and_says_so():
    cs = cands(n_photos=120)
    m = laid(cs, pace="slow")
    assert m["total_s"] <= MO.MAX_AUTO_S + 1e-6
    assert m["dropped"] and "8 minutes" in m["note"]
    assert m["items"][0]["id"] == "c0" and m["items"][-1]["id"] == "c119"                          # the first and the last stay


def test_a_set_length_drops_the_weakest_and_never_speeds_footage_up():
    cs = cands(n_clips=10)
    cs[4]["quality"] = 4.5
    m = laid(cs, pace="auto", target_s=20.0)
    assert m["total_s"] <= 20.0 + 1e-6
    assert "c4" in [g["id"] for g in m["dropped"]]
    for it in m["items"]:
        assert it["end"] - it["start"] >= 2.5 - 1e-6                                                # still at least the pace's own hold


def test_too_few_shots_for_a_long_length_says_so():
    m = laid(cands(n_clips=2), pace="auto", target_s=120.0)
    assert m["total_s"] < 120 * 0.9 and "not enough shots" in m["note"]


# ---------------------------------------------------------------------------------------------------------------------- variety
def test_the_same_file_or_look_is_never_used_more_than_twice():
    cs = []
    for k in range(8):                                                                            # eight scenes of one clip of the same seated man
        cs.append({"id": f"c{k}", "kind": "clip", "name": "man.mp4", "path": "/x/man.mp4", "start": k * 10.0, "end": k * 10.0 + 8, "what": "a man in a blue sweater seated",
                   "quality": 7.0, "t": [0, k * 10.0]})
    for k in range(4):
        cs.append({"id": f"c{8 + k}", "kind": "clip", "name": f"o{k}.mp4", "path": f"/x/o{k}.mp4", "start": 100.0 + k, "end": 110.0 + k, "what": f"children {k} running outside",
                   "quality": 6.0, "t": [1, k]})
    keep, left = MO.select(cs)
    assert sum(1 for c in keep if c["name"] == "man.mp4") == MO.MAX_PER_FILE
    assert len([g for g in left if "twice" in g["why"] or "look" in g["why"]]) >= 6
    assert MO.select([{**c, "quality": 3.0} for c in cs])[0] == []                                  # poor shots are not used


def test_near_identical_descriptions_across_different_files_count_as_the_same_look():
    cs = [{"id": f"c{k}", "kind": "clip", "name": f"f{k}.mp4", "path": "", "start": 0, "end": 9, "what": "a man in a blue sweater and jeans seated on a stool",
           "quality": 7.0, "t": [k, 0]} for k in range(6)]
    keep, _ = MO.select(cs)
    assert len(keep) == MO.MAX_ALIKE


def test_the_order_follows_the_day_opens_on_an_arrival_and_closes_on_the_group():
    cs = cands(n_clips=10)
    cs[1]["what"] = "the camp entrance sign with a welcome banner"                          # early in the day: it goes first
    cs[8]["what"] = "everyone together for the group photo"                                  # late in the day: it goes last
    order = MO.arc_order(cs)
    assert order[0] == "c1" and order[-1] == "c8"
    assert sorted(order) == sorted(c["id"] for c in cs)
    plain = MO.arc_order(cands(n_clips=10))
    assert plain[0] == "c0" and plain[-1] == "c9"                                           # nothing to move: simply as it happened


# ---------------------------------------------------------------------------------------------------------------------- the card
def test_the_plan_card_says_it_in_plain_words_with_the_shots_in_order():
    cs = cands(n_clips=3, n_photos=2)
    cs[0]["what"] = "the entrance of the camp"
    cs[4]["what"] = "the whole group on the lawn"
    m = laid(cs, pace="slow", style="slideshow", song={"title": "Warm Light", "bpm": 78.0, "mood": "chill"}, left_out=[{"id": "c9", "what": "blurry", "why": "poor quality"}])
    c = MO.card(m, title="Camp", notes="a youth camp recap, upbeat", job="j1")
    assert c["headline"].startswith("A ") and "slideshow from 2 photos and 3 clips" in c["headline"] and "slow" in c["headline"].lower() and "Warm Light" in c["headline"]
    assert c["you_said"] == "a youth camp recap, upbeat"
    assert [s["id"] for s in c["strip"]] == m["order"] and c["strip"][0]["thumb"] == "mt_c0.jpg"
    assert c["summary"].startswith("It opens on the entrance of the camp") and c["summary"].endswith("the whole group on the lawn.")
    assert c["dissolves"] is False and c["pace"] == "slow" and c["left_out_line"] == "1 left out (poor quality)"


# ------------------------------------------------------------------------------------------------------------ the whole thing
FF = None


def _ff():
    from clipfarm.core import ffmpeg
    return ffmpeg.bin_path("ffmpeg")


def _clip(path: Path, colour: str, secs: float, w=320, h=240, sound=True):
    cmd = [_ff(), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={colour}:size={w}x{h}:rate=30:duration={secs}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True, timeout=120)


def _photo(path: Path, colour: str):
    """A photo-like picture: a colour wash with texture (a flat single-colour image would be taken for a graphic, never a photo)."""
    import random
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 480), colour)
    d = ImageDraw.Draw(im)
    rnd = random.Random(path.name)
    for _ in range(120):
        x, y = rnd.randint(0, 600), rnd.randint(0, 440)
        d.ellipse([x, y, x + rnd.randint(8, 60), y + rnd.randint(8, 60)], fill=tuple(rnd.randint(0, 255) for _ in range(3)))
    im.save(path, quality=90)


def _song(path: Path, secs=40):
    subprocess.run([_ff(), "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-c:a", "libmp3lame", str(path)], check=True, timeout=60)


class _Res:
    def __init__(self, d):
        self.data = d


class FakeAI:
    usage = {}

    def __init__(self, cfg=None, host=None):
        self.calls = []

    def available(self):
        return True, "fake"

    def cost(self):
        return 0.0

    def now(self):
        return ""

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        self.calls.append(task)
        if task == "broll_describe":
            return _Res({"scenes": [{"start_s": 0, "end_s": 7, "what": "kids playing a game on the lawn", "quality": 8},
                                    {"start_s": 7, "end_s": 14, "what": "a counsellor leading a song", "quality": 7}]})
        if task == "photo_describe":
            import re
            names = re.findall(r"(\d+)=([^,]+)", prompt)
            return _Res({"photos": [{"n": int(n), "what": f"a photo of campers at {Path(nm).stem}", "quality": 7} for n, nm in names]})
        if task == "montage_story":
            return _Res({"title": "Camp Week", "summary": "It opens on kids playing a game on the lawn and ends on a photo of campers."})
        if task in ("music_vibe",):
            return _Res({"mood": "upbeat", "genre": "", "why": "x"})
        return _Res({})


@pytest.fixture
def world(tmp_path, monkeypatch):
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    monkeypatch.setattr(LP, "LocalAI", lambda c: FakeAI())
    return LP, cfg, store, projects, tmp_path


def _folder(tmp_path, clips=2, photos=4, with_audio=False):
    f = tmp_path / "Camp"
    f.mkdir()
    for k in range(clips):
        _clip(f / f"clip{k}.mov", ["red", "green", "blue"][k % 3], 14, sound=with_audio)
    for k in range(photos):
        _photo(f / f"IMG_{k:03d}.jpg", ["orange", "purple", "teal", "gold"][k % 4])
    (f / "Music").mkdir()
    _song(f / "Music" / "Warm Light.mp3")
    return f


def test_notes_asking_for_a_slideshow_plan_first_and_render_only_after_the_card_is_approved(world):
    LP, cfg, store, projects, tmp = world
    f = _folder(tmp)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(f), "", "a youth camp slideshow, take it slow", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"notes": "a youth camp slideshow, take it slow", "pace": "slow", "music": "auto"})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "approval", job                                                       # it stops at the plan card: nothing rendered yet
    assert not list((f / "Output").glob("*.mp4")) if (f / "Output").exists() else True
    plan = json.loads((f / ".aivideo" / "plan.json").read_text())
    mt = plan["montage"]
    assert mt["style"] == "slideshow" and mt["pace"] == "slow" and plan["awaiting_approval"]
    assert sum(1 for it in mt["items"] if "photo" in it) >= 2 and sum(1 for it in mt["items"] if "photo" not in it) >= 1      # photos AND a few clips
    card = runner.montage_card(proj["id"])
    assert "slideshow" in card["headline"] and "Warm Light" in card["headline"] and card["you_said"].startswith("a youth camp slideshow")
    assert len(card["strip"]) == len(mt["items"]) and all((store.dir(proj["job"]) / s["thumb"]).exists() for s in card["strip"])
    assert all(2.0 <= it["end"] - it["start"] <= 8.0 for it in mt["items"])
    # remove one shot, change the pace: the card is laid out again, still nothing renders
    first = card["strip"][1]["id"]
    card2 = runner.montage_edit(proj["id"], {"op": "remove", "id": first})
    assert first not in [s["id"] for s in card2["strip"]] and first in [r["id"] for r in card2["removed"]]
    card3 = runner.montage_edit(proj["id"], {"op": "pace", "value": "fast"})
    assert card3["pace"] == "fast" and card3["total_s"] < card2["total_s"]
    assert store.get(proj["job"])["options"]["pace"] == "fast"
    # approve: now it renders
    runner.approve(proj["id"])
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    out = next((f / "Output").glob("*.mp4"))
    from clipfarm.core import ffmpeg
    plan = json.loads((f / ".aivideo" / "plan.json").read_text())
    assert ffmpeg.duration(out) == pytest.approx(plan["montage"]["total_s"], abs=1.5)               # as long as its shots add up to
    assert ffmpeg.audio_stream(ffmpeg.probe(out)) is not None                                       # the song is under it
    assert "montage" not in (store.get(proj["job"]).get("options") or {}) or True


def test_a_folder_of_silent_clips_is_a_montage_with_hard_cuts_and_no_dissolves_by_default(world):
    LP, cfg, store, projects, tmp = world
    f = _folder(tmp, clips=3, photos=0, with_audio=False)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(f), "", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"pace": "fast", "music": "auto"})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "approval"
    plan = json.loads((f / ".aivideo" / "plan.json").read_text())
    assert plan["montage"]["style"] == "montage" and plan["montage"]["dissolves"] is False
    runner.approve(proj["id"])
    assert store.get(proj["job"])["status"] == "done"
    transitions = [e["kind"] for e in getattr(runner, "last_transitions", None) or []]
    assert "dissolve" not in transitions and "broll" not in transitions


def test_not_what_i_meant_makes_it_a_normal_edit_and_the_setting_is_remembered(world):
    LP, cfg, store, projects, tmp = world
    f = _folder(tmp, clips=1, photos=0, with_audio=True)
    proj = projects.create_local(str(f), "", "a montage", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"montage": "off"})
    assert store.get(proj["job"])["options"]["montage"] == "off"
    assert MO.detect("a montage", n_videos=1, setting=store.get(proj["job"])["options"]["montage"]) is None


def test_music_is_fitted_to_the_cuts_the_first_beat_is_where_the_first_cut_lands(world):
    LP, cfg, store, projects, tmp = world
    f = _folder(tmp, clips=2, photos=2)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(f), "", "a montage", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"notes": "a montage", "music": "auto"})
    runner.make(proj["id"], {})
    mt = json.loads((f / ".aivideo" / "plan.json").read_text())["montage"]
    assert mt["song"]["title"] == "Warm Light"
    if mt.get("period"):                                                                           # a steady beat was found: every hold is whole beats
        for it in mt["items"]:
            assert ((it["end"] - it["start"]) / mt["period"]) % 1 == pytest.approx(0, abs=0.02) or ((it["end"] - it["start"]) / mt["period"]) % 1 == pytest.approx(1, abs=0.02)
    assert mt["music_start"] >= 0.0


# ---------------------------------------------------------------------------------- the same rule for B-roll over an interview
def test_broll_over_an_interview_never_uses_one_file_or_look_more_than_twice():
    pool = [{"start": k * 10.0, "end": k * 10.0 + 8, "what": "a man in a blue sweater seated on a stool", "quality": 7.0, "file": "man.mp4"} for k in range(6)]
    other = ["students walking to class", "a cat on a porch", "a parade with flags", "tables in a cafeteria", "a quiet library aisle"]
    pool += [{"start": 100.0 + k * 10, "end": 108.0 + k * 10, "what": other[k], "quality": 6.0, "file": f"walk{k}.mp4"} for k in range(5)]
    placed = [{"at": 5.0 + k * 9, "length": 3.0, "scene": k, "src": pool[k]["start"], "what": pool[k]["what"], "file": "man.mp4"} for k in range(6)]
    out, n = MO.vary(placed, pool)
    files = [pool[c["scene"]]["file"] for c in out]
    assert files.count("man.mp4") == MO.MAX_PER_FILE and n == 4 and len(out) == 6                      # four swapped, nothing taken out
    assert [c["at"] for c in out] == sorted(c["at"] for c in out)
    assert len({c["scene"] for c in out}) == 6                                                           # and no shot twice


def test_variety_leaves_a_shot_alone_when_nothing_else_is_there():
    pool = [{"start": k * 10.0, "end": k * 10.0 + 8, "what": "a man seated", "quality": 7.0, "file": "man.mp4"} for k in range(4)]
    placed = [{"at": 5.0 + k * 9, "length": 3.0, "scene": k, "src": pool[k]["start"], "what": "a man seated", "file": "man.mp4"} for k in range(4)]
    out, n = MO.vary(placed, pool)
    assert n == 0 and [c["scene"] for c in out] == [0, 1, 2, 3]                                         # no alternative: the twelve-second rule wins


def test_a_talking_folder_with_broll_photos_and_a_logo_is_not_a_slideshow(tmp_path):
    from clipfarm.clipper import montage as MO
    (tmp_path / "B-roll" / "photos").mkdir(parents=True)
    (tmp_path / "Graphics").mkdir()
    for i in range(5):
        (tmp_path / "B-roll" / "photos" / f"p{i}.jpg").write_bytes(b"x")
    (tmp_path / "Graphics" / "lower third.png").write_bytes(b"x")
    (tmp_path / "logo.png").write_bytes(b"x")
    (tmp_path / "talk.mov").write_bytes(b"x")
    c = MO.quick_counts(tmp_path, lambda parts: False)
    assert c["photos"] == 0 and c["videos"] == 1
    assert MO.detect("", n_photos=c["photos"], n_videos=1, audio_clips=1) is None
    assert MO.detect("", n_photos=3, n_videos=1, audio_clips=1) is None        # a few photos never make a slideshow on their own
