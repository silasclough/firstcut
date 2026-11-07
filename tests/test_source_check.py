"""The engine looks at the SOURCE before it cuts: a still picture, the creator's own promo bars and captions, gameplay, a vague title (clipfarm/clipper/sourcecheck.py).

The gate (clipgate / opening) stays as the last look at the finished file; these tests are about not making the bad clip in the first place."""

import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import clipgate, moments as M, ocr, sourcecheck as SC
from clipfarm.clipper.engine import Engine
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config


def _ff(*args):
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *args], check=True)


def _still(tmp_path, secs=40):
    im = tmp_path / "s.png"
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (360, 640, 3), dtype=np.uint8)).save(im)
    out = tmp_path / "still.mp4"
    _ff("-loop", "1", "-framerate", "24", "-i", str(im), "-f", "lavfi", "-i", "sine=f=200", "-t", str(secs), "-pix_fmt", "yuv420p", "-shortest", str(out))
    return out


def _moving(tmp_path, secs=40, name="moving.mp4"):
    out = tmp_path / name
    _ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24", "-f", "lavfi", "-i", "sine=f=200", "-t", str(secs), "-pix_fmt", "yuv420p", "-shortest", str(out))
    return out


# ------------------------------------------------------------------ a picture that does not move
def test_motion_of_tells_a_still_picture_from_real_footage(tmp_path):
    assert SC.motion_of(_still(tmp_path, 12), 0, 10)["still"] is True
    assert SC.motion_of(_moving(tmp_path, 12), 0, 10)["still"] is False


def test_a_whole_source_that_is_one_picture_is_found_and_real_footage_is_not(tmp_path):
    r = SC.probe_source(_still(tmp_path))
    assert r["still"] and r["windows"] >= 3 and r["still_windows"] == r["windows"]
    assert SC.probe_source(_moving(tmp_path))["still"] is False


def test_a_short_file_is_not_judged(tmp_path):
    assert SC.probe_source(_still(tmp_path, 6)) == {"still": False, "windows": 0, "still_windows": 0}


def test_candidates_over_a_still_stretch_are_dropped_and_the_rest_kept(tmp_path, monkeypatch):
    src = tmp_path / "x.mp4"
    src.write_bytes(b"")
    good, bad, unseen = M.Moment(0, 20, 0, 3, score=8), M.Moment(30, 50, 4, 8, score=9), M.Moment(60, 80, 9, 12, score=2)
    monkeypatch.setattr(SC, "motion_of", lambda s, a, b: {"still": a == 30, "still_share": 0.97 if a == 30 else 0.0, "dur": b - a})
    kept, dropped = SC.screen_moments(src, [good, bad, unseen], limit=2)           # the third is beyond the limit: not looked at, kept
    assert kept == [good, unseen] and [m for m, _ in dropped] == [bad] and "does not move" in dropped[0][1]


def test_the_engine_refuses_a_still_source_before_transcribing(tmp_path):
    eng, store, job = _engine(tmp_path)
    src = _still(tmp_path)
    with pytest.raises(RuntimeError, match="one still picture"):
        eng._refuse_still_source(job["id"], src)
    eng._refuse_still_source(job["id"], _moving(tmp_path))                                # real footage: nothing happens
    eng.cfg.data.setdefault("clipper", {}).setdefault("source_check", {})["still"] = False
    eng._refuse_still_source(job["id"], src)                                              # switched off: nothing happens


def test_screen_stills_picks_the_next_moment_and_fails_clearly_when_nothing_moves(tmp_path, monkeypatch):
    eng, store, job = _engine(tmp_path)
    a, b = M.Moment(0, 20, 0, 3, title="a", score=7), M.Moment(30, 50, 4, 8, title="b", score=9)
    monkeypatch.setattr(SC, "motion_of", lambda s, x, y: {"still": x == 30, "still_share": 1.0, "dur": y - x})
    assert eng._screen_stills(job["id"], tmp_path / "x.mp4", [a, b]) == [a]
    monkeypatch.setattr(SC, "motion_of", lambda s, x, y: {"still": True, "still_share": 1.0, "dur": y - x})
    with pytest.raises(RuntimeError, match="still picture"):
        eng._screen_stills(job["id"], tmp_path / "x.mp4", [a, b])


# ------------------------------------------------------------------ text burned into the source
def L(t, y, x=0.1, w=0.3, h=0.04, c=0.9):
    return {"t": t, "x": x, "y": y, "w": w, "h": h, "c": c}


def test_a_subscribe_bar_a_handle_and_a_phone_number_are_found_with_how_much_to_crop():
    pages = [[L("SUBSCRIBE", 0.90)], [L("@officialandy", 0.93), L("call 210 555 0254", 0.05)], [L("SUBSCRIBE", 0.90)], [], [], []]
    f = SC.analyse(pages, "so you build a curriculum")
    assert "SUBSCRIBE" in f["promo"] and "@officialandy" in f["promo"]
    assert f["bottom"] == pytest.approx(0.10 + 0.012, abs=0.02)          # the bar starts at 90%: cut the bottom tenth
    assert f["top"] == pytest.approx(0.09 + 0.012, abs=0.02)             # the number sits at the very top
    plan = SC.crop_plan(f)
    assert plan["top"] > 0 and plan["bottom"] > 0 and plan["own_captions"] is True


def test_a_name_tag_that_stays_put_is_cropped_but_a_line_that_is_said_aloud_is_not():
    tag = [[L("Andy Elliott - Creator of Project 500", 0.84)]] * 5
    assert SC.analyse(tag, "")["bottom"] > 0.15
    said = [[L("subscribe to the channel", 0.88)]] * 5
    assert SC.analyse(said, "he said subscribe to the channel")["bottom"] == 0.0           # the speaker's own sentence is not a banner


def test_nothing_burned_in_means_nothing_cropped():
    f = SC.analyse([[L("hello there", 0.5)], [], [], [], [], []], "hello there")
    assert f["promo"] == [] and f["top"] == 0.0 and f["bottom"] == 0.0 and f["captions"] is False
    assert SC.crop_plan(f) == {"top": 0.0, "bottom": 0.0, "own_captions": True}
    assert SC.crop_plan(None) == {"top": 0.0, "bottom": 0.0, "own_captions": True}


def test_the_sources_own_bottom_captions_are_cropped_away_and_ours_stay():
    words = ["little bit of", "I always say", "when you're building", "brick matters", "remember that", "every brick counts"]
    pages = [[L(t, 0.88, x=0.3, w=0.4)] for t in words]
    f = SC.analyse(pages, " ".join(words))
    assert f["captions"] is True and f["cap_band"][0] == pytest.approx(0.88)
    plan = SC.crop_plan(f)
    assert plan["bottom"] > 0.1 and plan["own_captions"] is True


def test_the_sources_captions_in_the_middle_cannot_be_cropped_so_ours_are_left_off():
    words = ["little bit of", "I always say", "when you're building", "brick matters", "remember that", "every brick counts"]
    f = SC.analyse([[L(t, 0.55, x=0.3, w=0.4)] for t in words], " ".join(words))
    plan = SC.crop_plan(f)
    assert f["captions"] is True and plan["own_captions"] is False and plan["bottom"] == 0.0


def test_a_crop_that_would_eat_the_picture_is_not_made():
    plan = SC.crop_plan({"promo": ["x"], "top": 0.2, "bottom": 0.3, "captions": False, "cap_band": None})
    assert plan["top"] == 0.0 and plan["bottom"] == 0.0


def test_crop_filter_gives_even_sizes_and_the_new_height():
    f, h, y0 = SC.crop_filter(720, {"top": 0.1, "bottom": 0.2})
    assert f == f",crop=iw:{h}:0:{y0}" and h == 504 and y0 == 72 and h % 2 == 0 and y0 % 2 == 0
    assert SC.crop_filter(720, {"top": 0.0, "bottom": 0.0}) == ("", 720, 0)


def test_no_text_reader_means_not_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "read", lambda shots: None)
    assert SC.burned_in(_moving(tmp_path, 12), 0, 10) is None


def test_burned_in_reads_frames_of_the_stretch(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ocr, "read", lambda shots: seen.setdefault("n", len(shots)) and [[L("SUBSCRIBE", 0.91)]] * len(shots))
    f = SC.burned_in(_moving(tmp_path, 12), 0, 10, "")
    assert seen["n"] == SC.FRAMES and f["bottom"] > 0.08


# ------------------------------------------------------------------ the render crops it out
def _engine(tmp_path):
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {})
    return Engine(cfg, store), store, job


def _red_share(video, t=1.0):
    raw = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-vf", "scale=108:192", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(int)
    return float(((px[:, 0] > 200) & (px[:, 1] < 60) & (px[:, 2] < 60)).mean())


def _render(tmp_path, monkeypatch, found, style="none"):
    from clipfarm.clipper import reframe as R
    eng, store, job = _engine(tmp_path)
    src = store.dir(job["id"]) / "source.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x3070a0:s=1280x720:r=25:d=4", "-f", "lavfi", "-i", "sine=f=300:d=4", "-vf", "drawbox=x=0:y=612:w=1280:h=108:color=red:t=fill",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src))               # a red bar across the bottom 15% stands for "SUBSCRIBE"
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([[]] * 8, 8))                                 # no faces: the whole frame is shown (mode "fit")
    monkeypatch.setattr(SC, "burned_in", lambda *a, **k: found)
    words = [{"text": f"w{i}", "start": 0.3 + i * 0.4, "end": 0.6 + i * 0.4} for i in range(8)]
    m = M.Moment(start=0.2, end=3.6, first=0, last=7, title="T", caption="c", score=8)
    clip = eng.render_clip(job["id"], 1, m, words, src, layout="fit", style=style, pace="natural", hook_title=False)
    return clip, store.dir(job["id"]) / clip["file"], store.dir(job["id"])


def test_the_promo_bar_is_cropped_out_of_the_finished_clip(tmp_path, monkeypatch):
    clip, out, d = _render(tmp_path, monkeypatch, {"promo": ["SUBSCRIBE"], "top": 0.0, "bottom": 0.17, "captions": False, "cap_band": None})
    assert clip["source_fix"]["bottom"] == pytest.approx(0.17) and _red_share(out) == 0.0
    assert json.loads((d / "clip_01_srccheck.json").read_text())["plan"]["bottom"] == pytest.approx(0.17)      # read once, kept next to the clip


def test_without_a_finding_the_picture_is_left_alone(tmp_path, monkeypatch):
    clip, out, _ = _render(tmp_path, monkeypatch, {"promo": [], "top": 0.0, "bottom": 0.0, "captions": False, "cap_band": None})
    assert "source_fix" not in clip and _red_share(out) > 0.05


def test_the_sources_own_middle_captions_switch_ours_off(tmp_path, monkeypatch):
    clip, _, _ = _render(tmp_path, monkeypatch, {"promo": [], "top": 0.0, "bottom": 0.0, "captions": True, "cap_band": (0.5, 0.58)}, style="bold")
    assert clip["caption_style"] == "none" and clip["source_fix"]["own_captions"] is False


def test_a_text_reader_that_breaks_does_not_stop_the_render(tmp_path, monkeypatch):
    from clipfarm.clipper import reframe as R
    eng, store, job = _engine(tmp_path)
    monkeypatch.setattr(SC, "burned_in", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no reader")))
    assert eng._strip_plan(store.dir(job["id"]), 1, tmp_path / "x.mp4", 0.0, 4.0, []) == {"top": 0.0, "bottom": 0.0, "own_captions": True}
    assert R  # (imported for the monkeypatch in the sibling tests)


# ------------------------------------------------------------------ gameplay
@pytest.mark.parametrize("info, hit", [
    ({"categories": ["Gaming"], "title": "xQc plays"}, True),
    ({"categories": ["Entertainment"], "title": "NoPixel heist goes wrong"}, True),
    ({"categories": ["People & Blogs"], "title": "Day in my life", "tags": ["vlog", "minecraft"]}, True),
    ({"categories": ["Education"], "title": "How The Office almost died"}, False),
    ({"categories": ["News & Politics"], "title": "The great train robbery heist of 1963"}, False),
    ({}, False)])
def test_youtube_gameplay_is_told_from_real_life(info, hit):
    assert bool(SC.gameplay_meta(info)) is hit


def test_a_twitch_clip_made_in_a_game_is_gameplay_but_just_chatting_and_unknown_are_not():
    assert SC.gameplay_twitch({"game_id": "32982", "game_name": "Grand Theft Auto V"})
    assert SC.gameplay_twitch({"game_id": "509658"}) is None                # Just Chatting
    assert SC.gameplay_twitch({"game_id": "509672"}) is None                # IRL
    assert SC.gameplay_twitch({}) is None and SC.gameplay_twitch({"game_id": ""}) is None


def test_twitch_candidates_skip_gameplay_clips_unless_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TWITCH_CLIENT_ID", "id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "secret")
    from test_twitch_clips import Cfg, FakeHTTP, clip
    from clipfarm.clipper.twitch import Twitch
    gta, chat = {**clip("a", 5000), "game_id": "32982"}, {**clip("b", 4000), "game_id": "509658"}
    for opts, want in (({}, ["tw_b"]), ({"skip_gameplay": False}, ["tw_a", "tw_b"])):
        tw = Twitch(Cfg(tmp_path, [{"name": "Kai Cenat"}], **opts), tmp_path, http=FakeHTTP({"1": [gta, chat]}))
        got = sorted(c["id"] for c in tw.candidates(lambda cid: False, lambda login: 0))
        assert got == want


def test_discovery_marks_a_gaming_category_video_as_avoided(tmp_path, monkeypatch):
    from clipfarm.clipper import discover
    seen = {}
    monkeypatch.setattr(discover, "details", lambda url: {"title": "Funniest moments ever", "duration": 900, "categories": ["Gaming"], "channel": "X", "id": "v1"})
    f = discover.Finder.__new__(discover.Finder)
    f.acfg, f.state, f.taste = {"min_minutes": 1, "avoid_words": []}, type("S", (), {"data": {"seen": {}}, "save": lambda self: None})(), None
    f.seen = lambda vid: False
    f.mark = lambda vid, why: seen.setdefault(vid, why)
    assert f._accept({"id": "v1", "title": "Funniest moments ever"}, "cc", "test") is None
    assert "gaming" in seen["v1"]


# ------------------------------------------------------------------ titles
class FakeLLM:
    def __init__(self, titles):
        self.titles, self.calls = titles, []

    def complete_json(self, task, system, prompt, schema):
        self.calls.append((task, prompt))

        class R:
            data = {"titles": [{"clip": k, "title": t} for k, t in self.titles.items()]}
        return R()


def _sents(n):
    return [M.Sentence(i, i * 5.0, i * 5.0 + 4.8, f"he said number {i} about the Chernobyl reactor") for i in range(n)]


def test_a_vague_title_is_asked_for_again_once_and_a_good_answer_replaces_it():
    ms = [M.Moment(0, 20, 0, 3, title="How do I do that?", score=7), M.Moment(30, 50, 4, 8, title="Chernobyl's control rods were pulled out", score=7)]
    llm = FakeLLM({0: "What the operators did to Chernobyl's reactor at 1:23 AM"})
    M.retitle_weak(llm, ms, _sents(10))
    assert ms[0].title.startswith("What the operators did") and ms[1].title == "Chernobyl's control rods were pulled out"
    assert len(llm.calls) == 1 and llm.calls[0][0] == "clip_title" and "refused" in llm.calls[0][1] and "Chernobyl's control" not in llm.calls[0][1]


def test_a_second_vague_answer_is_not_taken_and_a_failing_ai_changes_nothing():
    ms = [M.Moment(0, 20, 0, 3, title="How do I do that?", score=7)]
    M.retitle_weak(FakeLLM({0: "Wow what a moment"}), ms, _sents(10))
    assert ms[0].title == "How do I do that?"
    boom = type("B", (), {"complete_json": lambda self, *a: (_ for _ in ()).throw(RuntimeError("down"))})()
    M.retitle_weak(boom, ms, _sents(10))
    assert ms[0].title == "How do I do that?" and clipgate.weak_title(ms[0].title)
    assert M.retitle_weak(FakeLLM({}), [M.Moment(0, 20, 0, 3, title="Why the comedy industry doesn't want real people")], _sents(10))     # nothing weak: no call, same list


def test_a_twitch_clip_with_a_vague_title_gets_one_more_ask(tmp_path, monkeypatch):
    eng, store, job = _engine(tmp_path)
    src = _moving(tmp_path, 6)
    answers = iter(["How do I do that?", "xQc explains why every brick matters in a wall"])

    class Router:
        def complete_json(self, task, system, prompt, schema):
            class R:
                data = {"title": next(answers), "caption": "c", "hashtags": ["xqc"]}
            return R()
    monkeypatch.setattr(Engine, "router_for_now", lambda self: (Router(), "test"))
    words = [{"text": "every", "start": 0.5, "end": 0.9}, {"text": "brick", "start": 1.0, "end": 1.4}]
    m = eng._whole_clip_moment({"source": {"title": "lol", "channel": "xQc", "views": 900}}, words, src)[0]
    assert m.title == "xQc explains why every brick matters in a wall"
