"""The long-form editor's accuracy fixes of 2026-10-13, one test each (found by checking real projects: Shoe Shack, AI Test, AI Vid 2)."""

import json
import subprocess
from fractions import Fraction

import pytest

from clipfarm.clipper import moments as M


class _Res:
    def __init__(self, d):
        self.data = d


# ----------------------------------------------------------------------------- punctuation and sentences
def test_whisper_words_get_the_punctuation_of_its_own_segment_text():
    from clipfarm.clipper.openai_audio import has_punctuation, punctuate
    words = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate("so we moved here in may it was hard".split())]
    segs = [{"text": " So we moved here in May."}, {"text": " It was hard!"}]
    got = punctuate(words, segs)
    assert [w["text"] for w in got] == ["So", "we", "moved", "here", "in", "May.", "It", "was", "hard!"]
    assert [(w["start"], w["end"]) for w in got] == [(w["start"], w["end"]) for w in words]       # the times are untouched
    assert punctuate(words, []) == words
    assert not has_punctuation(words * 4) and has_punctuation(got * 4)


def test_chunks_overlap_and_a_boundary_word_is_kept_exactly_once():
    from clipfarm.clipper.openai_audio import chunk_windows, keep_share
    wins = chunk_windows(2500.0, 1200.0, 3.0)
    assert wins[0] == (0.0, 1203.0, 0.0, 1200.0) and wins[1][0] == 1197.0 and wins[-1][3] == float("inf")
    word = {"text": "boundary", "start": 1199.7, "end": 1200.4}                       # its middle is at 1200.05: the second chunk's
    first = keep_share([{"text": "boundary", "start": word["start"], "end": word["end"]}], 0.0, 0.0, 1200.0)
    second = keep_share([{"text": "boundary", "start": word["start"] - 1197.0, "end": word["end"] - 1197.0}], 1197.0, 1200.0, 2400.0)
    assert first == [] and [w["text"] for w in second] == ["boundary"]


def _words(text, t0=0.0, step=0.3, gap_after=None):
    out, t = [], t0
    for k, w in enumerate(text.split()):
        out.append({"text": w, "start": round(t, 3), "end": round(t + 0.25, 3)})
        t += step + (gap_after.get(k, 0.0) if gap_after else 0.0)
    return out


def test_with_punctuation_a_long_sentence_stays_whole_without_it_the_old_rule_stands():
    from clipfarm.clipper import localproj as LP
    long = " ".join(f"w{i}" for i in range(28)) + "."
    punct = _words(long + " " + " ".join(f"x{i}" for i in range(10)) + ".")
    s = LP.seg(punct)
    assert len(s) == 2 and s[0].text.endswith("w27.")                                  # 28 words, one sentence
    plain = _words(" ".join(f"w{i}" for i in range(40)))
    assert [len(x.text.split()) for x in LP.seg(plain)] == [22, 18]                    # old transcripts: exactly as before


def test_the_video_never_ends_on_half_a_sentence():
    from clipfarm.clipper import localproj as LP
    # with punctuation: run on to the speaker's own sentence end
    sents = [M.Sentence(0, 0, 2, "We moved in May."), M.Sentence(1, 2.5, 4, "And then the car broke"), M.Sentence(2, 4.1, 5, "down on the highway."),
             M.Sentence(3, 6, 7, "Okay."), M.Sentence(4, 8, 9, "Next.")]
    items = LP.finish_sentence([{"kind": "talk", "a": 0, "b": 1}], sents)
    assert items[0]["b"] == 2
    # without punctuation (older transcripts): the 22-word limit split "... going to get | her" - the next words follow with no pause
    sents = [M.Sentence(0, 0, 5, "I was like oh this is going to get"), M.Sentence(1, 5.05, 8, "her She liked it a lot"),
             M.Sentence(2, 9.5, 11, "Then we went home"), M.Sentence(3, 13, 14, "x"), M.Sentence(4, 15, 16, "y")]
    items = LP.finish_sentence([{"kind": "talk", "a": 0, "b": 0}], sents)
    assert items[0]["b"] == 1                                                           # runs on to "her ..." but not across the pause
    # every item finishes its thought (not only the last one) when asked
    sents = [M.Sentence(0, 0, 2, "We moved in May."), M.Sentence(1, 2.5, 4, "And then the car broke"), M.Sentence(2, 4.1, 5, "down on the highway."),
             M.Sentence(3, 20, 22, "Next we found a house."), M.Sentence(4, 22.5, 24, "It had a red door."), M.Sentence(5, 30, 31, "The end.")]
    both = LP.finish_sentence([{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 3, "b": 4}], sents, every=True)
    assert both[0]["b"] == 2 and both[1]["b"] == 4


# ----------------------------------------------------------------------------- cut points
def test_a_kept_stretch_never_reaches_into_the_next_word_even_the_interviewers():
    from clipfarm.clipper import localproj as LP
    guest = _words("We fit every foot by hand.", 10.0) + _words("So the next thing", 12.4)            # "So" starts 0.6 s after "hand."
    host = [{"text": "Mm-hmm.", "start": 11.85, "end": 12.1}]                                        # the interviewer, in between
    all_words = sorted(guest + host, key=lambda w: w["start"])
    sents = LP.seg(guest)
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    (a, b), = LP.LocalRunner.intervals_of(r, {"items": [{"kind": "talk", "a": 0, "b": 0}], "gap": 0.75}, sents, guest, all_words)
    assert b <= 11.85 - 0.04 + 1e-6                                                     # stops before the interviewer's "Mm-hmm"
    assert b >= guest[5]["end"]                                                         # and after "hand." itself
    [(a2, b2)], bounds = LP.LocalRunner.intervals_of(r, {"items": [{"kind": "talk", "a": 0, "b": 0}], "gap": 0.75}, sents, guest, all_words,
                                                     with_bounds=True)
    assert (a2, b2) == (a, b) and bounds[0][1] == pytest.approx(11.81)


def test_tightening_follows_the_sound_but_never_past_the_limits():
    import numpy as np

    from clipfarm.clipper import tighten as TI
    e = np.full(600, -60.0, dtype=np.float32)
    e[100:290] = -20.0                                          # 1.0 s to 2.9 s: this word runs straight on into the next one (no silence)

    class Meter:
        def curve(self, path):
            return e, TI._threshold(np.concatenate([e, np.full(600, -60.0, dtype=np.float32)]))
    free = TI.refine([(1.0, 2.5)], lambda t: ("x.wav", t), Meter())
    assert free[0][1] > 2.9                                    # without a limit the cut runs on through the next word
    out = TI.refine([(1.0, 2.5)], lambda t: ("x.wav", t), Meter(), bounds=[(0.9, 2.6)])
    assert out[0][1] <= 2.6 + 1e-6 and out[0][0] >= 0.9       # with it, it stops before the next word starts


# ----------------------------------------------------------------------------- frame rate and timecode
def test_timecode_is_frames_at_the_nominal_rate_times_the_real_frame_length():
    from clipfarm.clipper import localproj as LP
    info = {"streams": [{"codec_tag_string": "tmcd", "tags": {"timecode": "03:51:50:08"}}]}
    assert LP.timecode_s(info, Fraction(24000, 1001)) == pytest.approx(13924.244, abs=1e-3)       # not 13910.333
    assert LP.timecode_s(info, Fraction(24)) == pytest.approx(13910.3333, abs=1e-3)
    df = {"streams": [{"codec_tag_string": "tmcd", "tags": {"timecode": "01:00:00;00"}}]}                                        # 29.97 drop-frame
    assert LP.timecode_df(df) and LP.timecode_s(df, Fraction(30000, 1001)) == pytest.approx(3599.9964, abs=1e-3)
    assert LP.timecode_s({"format": {}}, 24.0) == 0.0


def test_the_timeline_keeps_the_cameras_rate_and_pieces_are_whole_frames():
    from clipfarm.clipper import fcpxml as FX
    from clipfarm.clipper import localproj as LP
    assert LP.timeline_rate(23.976023976023978) == Fraction(24000, 1001)              # was forced to 24
    assert LP.timeline_rate(29.97) == Fraction(30000, 1001) and LP.timeline_rate(119.88) == Fraction(60000, 1001)
    assert FX.std_rate(round(23.976023976, 3)) == Fraction(24000, 1001)                  # was 2997/125
    parts = [{"d": 2.0, "v": {"path": "v", "ss": 0}, "a": None, "item": {"kind": "clip"}}, {"d": 1.0, "v": {"path": "v", "ss": 5}, "a": None, "item": {"kind": "clip"}}]
    q = LP.quantize(parts, Fraction(24000, 1001))
    assert [p["n"] for p in q] == [48, 24] and q[0]["d"] == pytest.approx(48 * 1001 / 24000)


# ----------------------------------------------------------------------------- the style rules
def _projects(tmp_path, rules_text=None):
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI"), "local": True, "data_dir": "ai"}
    if rules_text is not None:
        (tmp_path / "ai").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ai" / "style_rules.md").write_text(rules_text)

    class Eng:
        def creative_router(self):
            class R:
                def complete_json(self, task, system, prompt, schema, **kw):
                    return _Res({"rules": ["Always open on the strongest line"]})
            return R()
    return AV, AV.Projects(cfg, Store(tmp_path / "jobs"), Eng())


def test_rules_cut_off_by_the_old_220_character_limit_are_put_back_whole(tmp_path):
    from clipfarm.clipper import aivideo as AV
    multicam = next(r for r in AV.DEFAULT_RULES if r.startswith("A multicam shoot"))
    cut = multicam[:220]
    AV_, p = _projects(tmp_path, f"# Editing style rules\n\n- {cut}\n- My own rule\n")
    text = p.rules()
    assert multicam in text and "never the sound of whichever camera is showing" in text and "My own rule" in text
    assert text.count("A multicam shoot") == 1


def test_a_rule_you_delete_or_rewrite_on_the_page_stays_that_way(tmp_path):
    AV, p = _projects(tmp_path)
    ads = next(r for r in AV.DEFAULT_RULES if r.startswith("ADVERTISEMENTS: use energetic"))
    text = p.rules().replace(f"- {ads}\n", "") + "- Never use a film-strip border on anything.\n"
    p.rules_path().write_text(text)                                                     # you deleted one rule and wrote your own
    AV.Projects(p.cfg, p.store, p.engine)                                               # the app restarts
    after = p.rules()
    assert ads not in after and "Never use a film-strip border on anything." in after


def test_the_editor_reads_whole_rules_and_old_learned_ones_make_room_first(tmp_path):
    AV, p = _projects(tmp_path)
    p.add_rules([f"Learned rule number {i} " + "x" * 300 for i in range(60)])
    text = p.rules_for_brief(budget=8000)
    assert all(r in text for r in AV.DEFAULT_RULES)                                     # every taught rule, whole
    assert "Learned rule number 59" in text and "Learned rule number 0 " not in text    # the newest learned ones fit, the oldest went
    assert all(ln.startswith("- ") for ln in text.splitlines())                         # whole lines only


def test_rules_are_learned_with_the_ai_videos_key_never_the_macs_own_model(tmp_path, monkeypatch):
    from clipfarm.clipper import localai as LAI
    AV, p = _projects(tmp_path)
    p.cfg.data["clipper"]["ai_videos"]["local_projects"] = True
    used = []

    class Fake:
        def __init__(self, cfg, host=None):
            pass

        def available(self):
            return True, "fake"

        def complete_json(self, task, system, prompt, schema, **kw):
            used.append((task, "CHANGE" in system or "change" in prompt))
            return _Res({"rules": []})
    monkeypatch.setattr(LAI, "LocalAI", Fake)
    p.learn("slightly too long, cut 20 seconds", "t", fallback=False, change=True)
    assert used == [("ai_style_rule", True)]


# ----------------------------------------------------------------------------- the edit
def test_silent_shots_can_be_picked_again():
    from clipfarm.clipper import broll as B
    seen = {}

    class Llm:
        def complete_json(self, task, system, prompt, schema, **kw):
            seen["schema"] = schema
            return _Res({"keep": [1], "why": "x"})
    scenes = [{"what": "a", "start": 0, "end": 4, "quality": 8}, {"what": "b", "start": 10, "end": 14, "quality": 8}]
    assert B.pick_standalone(Llm(), scenes, "", "", 20.0) == [1]
    assert "keep" in seen["schema"]["properties"]


def test_a_guest_quoting_someone_is_not_mistaken_for_the_interviewer():
    from clipfarm.clipper import speakers as SP
    assert not SP.COACH.search("My doctor said tell me what's going on and I just cried")
    assert not SP.COACH.search("and I thought how did you even do that")
    assert SP.COACH.search("Okay, tell me about the day you moved")
    assert SP.COACH.search("Could you say that again?")


def test_longer_means_the_kept_passages_grow_in_place():
    from clipfarm.clipper import localproj as LP
    lines = ["We opened the shop in May", "Rent was higher than we thought", "Our first customer was a nurse", "She stood all day long",
             "Her feet hurt every evening", "We measured both feet carefully", "Then we found a wider shoe", "She came back a week later",
             "She brought three friends along", "Now they all shop here", "Word travels fast in town", "We still remember her name"]
    sents = [M.Sentence(i, i * 3.0, i * 3.0 + 2.5, lines[i]) for i in range(12)]
    words = [{"text": "w", "start": s.start + k * 0.5, "end": s.start + k * 0.5 + 0.4} for s in sents for k in range(5)]
    plan = {"items": [{"kind": "talk", "a": 2, "b": 3}, {"kind": "talk", "a": 7, "b": 8}], "gap": 0.75}
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    out = LP.LocalRunner.grow(r, plan, sents, words, None, 22.0, None, None)
    a = [(it["a"], it["b"]) for it in out["items"]]
    assert len(a) == 2 and a[0][0] <= 2 and a[0][1] >= 4 and a[1][0] <= 7 and a[1][1] >= 9      # both grew, next to themselves
    assert a[0] < a[1]                                                                             # nothing tacked on at the end


def test_a_repeated_line_inside_a_longer_passage_is_cut_out():
    from clipfarm.clipper import localproj as LP
    sents = [M.Sentence(0, 0, 3, "We fit every single foot by hand here."), M.Sentence(1, 4, 6, "It takes time."),
             M.Sentence(2, 7, 10, "And honestly we fit every single foot by hand here."), M.Sentence(3, 11, 13, "People notice.")]
    out = LP.no_repeats([{"kind": "talk", "a": 0, "b": 0}, {"kind": "talk", "a": 1, "b": 3}], sents)
    assert [(it["a"], it["b"]) for it in out] == [(1, 3)]                                    # one take stays: the best one (complete, clean, the later), not the first


def test_the_transcript_comes_from_the_cleanest_camera():
    from clipfarm.clipper import multicam as MC
    a = {"path": "A.mov", "t0": 0.0, "dur": 100.0}
    b = {"path": "B.mov", "t0": 5.0, "dur": 85.0}
    group = {"angles": [{"id": "A", "files": [a]}, {"id": "B", "files": [b]}], "recorders": [], "dur": 100.0,
             "sound_q": {"A.mov": 33.0, "B.mov": 17.6}}                                       # like AI Test: A is the cleaner camera
    spans = MC.sound_spans(group)
    assert [(c["path"], x0, x1) for c, x0, x1 in spans] == [("A.mov", 0.0, 100.0)]
    group["sound_q"] = {"A.mov": 10.0, "B.mov": 30.0}
    assert [c["path"] for c, *_ in MC.sound_spans(group)] == ["A.mov", "B.mov", "A.mov"]     # B where it covers, A before and after
    rec = {"path": "rec.wav", "t0": 0.0, "dur": 99.9996}                                     # a camera running 0.4 ms past the recorder
    group["recorders"] = [rec]
    assert [c["path"] for c, *_ in MC.sound_spans(group)] == ["rec.wav"]                     # that sliver is not transcribed


def test_what_kind_of_video_the_notes_ask_for():
    from clipfarm.clipper import localproj as LP

    class AI:
        def __init__(self, kind):
            self.kind = kind

        def complete_json(self, *a, **k):
            if self.kind is None:
                raise RuntimeError("offline")
            return _Res({"kind": self.kind})
    assert LP.video_kind(AI("story"), "Don't make this feel like an ad.", "auto") == "story"     # the AI reads the meaning
    assert LP.video_kind(AI("story"), "anything", "ad") == "ad"                                 # your choice on the form wins
    assert LP.video_kind(AI(None), "A 1-minute ad for the shop", None) == "ad"                  # no AI: the keyword guess
    assert LP.video_kind(AI("promo"), "", None) == "story"


def test_a_song_shorter_than_the_video_repeats_with_a_crossfade():
    from clipfarm.clipper import music as MU
    assert MU.loop_plan(180.0, 10.0, 60.0) == [(0.0, 10.0, 60.0)]
    plan = MU.loop_plan(100.0, 20.0, 250.0, 2.0)
    assert plan[0] == (0.0, 20.0, 80.0) and plan[1] == (78.0, 0.0, 100.0)
    assert plan[-1][0] + plan[-1][2] == pytest.approx(250.0)


# ----------------------------------------------------------------------------- Final Cut
def test_a_single_camera_project_gets_real_clips_of_the_originals_with_the_recorders_sound(tmp_path):
    from clipfarm.clipper import fcpxml as FX
    items = [{"kind": "clip", "out": 0, "dur": 5.0, "path": "/Volumes/X/Shoot/C0001.MP4", "start": 12.0, "w": 3840, "h": 2160,
              "fps": 23.976023976023978, "audio": True, "file_dur": 600.0, "tc": 13924.244, "tc_df": False,
              "aud": {"path": "/Volumes/X/Shoot/rec.WAV", "start": 14.5, "tc": 0.0, "dur": 700.0, "ch": 1}},
             {"kind": "clip", "out": 5.0, "dur": 4.0, "path": "/Volumes/X/Shoot/C0001.MP4", "start": 40.0, "w": 3840, "h": 2160,
              "fps": 23.976023976023978, "audio": True, "file_dur": 600.0, "tc": 13924.244}]
    music = FX.MusicBed("/Volumes/X/Music/song.mp3", 6.0, "song", start=0.0, keyframes=[(0, -14), (2, -24)], crossfade=1.0)
    xml = FX.build_multicam("Talk", 1920, 1080, Fraction(24000, 1001), None, items, music=music,
                            titles=[{"at": 0.0, "len": 3.2, "text": "Find Your Fit", "kind": "hook"}])
    assert "<multicam" not in xml and xml.count('<asset-clip ref="a1"') == 2
    assert 'frameDuration="1001/24000s"' in xml and "2997/125" not in xml
    assert 'adjust-volume amount="-96dB"' in xml and 'lane="-3"' in xml and "rec" in xml            # the recorder carries the sound
    assert xml.count('name="Music: song') == 2                                                     # 9 s of video, a 6 s song: it repeats
    p = tmp_path / "t.fcpxml"
    p.write_text(xml)
    if FX.DTD_DIR.exists():
        assert FX.validate(p) == []


def test_graphics_titles_sit_where_the_video_shows_them():
    from clipfarm.clipper import fcpxml as FX
    (x, y), size = FX.graphic_position("Find Your Fit", "hook", 1920, 1080)
    assert x < 0 and y < 0                                                              # lower left, like the video (it used to be top left)


def test_a_still_that_is_in_the_footage_is_not_a_frozen_render(tmp_path):
    from clipfarm.clipper import localproj as LP
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    still = tmp_path / "still.mp4"                                                      # 4 s of moving picture, 4 s of a still photo, 4 s moving
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=4", "-f", "lavfi", "-i",
                    "color=c=blue:size=320x240:rate=24:duration=4", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=4", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=12", "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1[v]", "-map", "[v]", "-map", "3:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(still)], check=True, timeout=120)
    assert any("freezes" in x for x in LP.check_video(still, 12.0))                     # on its own it looks frozen...
    assert LP.check_video(still, 12.0, reference=still) == []                           # ...but the rough cut has the same still: fine


def test_a_change_to_a_scripted_video_rechecks_the_script():
    from clipfarm.clipper import flow as FL
    sents = [M.Sentence(0, 0, 3, "I was 22 and in design school."), M.Sentence(1, 4, 6, "Then Hannah from Little Steps called.")]

    class AI:
        def complete_json(self, *a, **k):
            return _Res({"steps": []})
    rep = FL.recheck(AI(), sents, [{"kind": "talk", "a": 1, "b": 1, "step": 2}], ["Pregnancy: 22, design school", "Harbor: Hannah, Little Steps"])
    assert rep[0]["missing"] == ["22"] and rep[1]["missing"] == [] and rep[1]["covered"] and rep[0]["rechecked"]


def test_script_accuracy_helpers_keep_whole_thoughts_names_numbers_and_rendered_proof():
    from clipfarm.clipper import flow as FL
    sents = [M.Sentence(0, 0, 1, "Jolene helped us."), M.Sentence(1, 1, 2, "And she found twenty two pairs."),
             M.Sentence(2, 2, 3, "That changed everything.")]
    assert FL.whole_thought(sents, 1) == (0, 1)
    assert FL.heard("22", "she found twenty two pairs")
    corrected, fixes = FL.correct_names([{"text": "Eolene", "start": 0.0, "end": 0.2}], ["Jolene helped us", "A", "B"])
    assert corrected[0]["text"] == "Jolene" and fixes
    mapped = FL.script_map(sents, [{"kind": "talk", "a": 0, "b": 2, "step": 1}], ["Jolene's result"])
    assert [x["text"] for x in mapped[0]["passages"][0]["lines"]] == [x.text for x in sents]

    class AI:
        def complete_json(self, task, *args, **kwargs):
            assert task == "flow_final"
            return _Res({"steps": [{"n": 1, "missing": [], "quotes": ["Jolene helped us"]}]})
    assert FL.final_check(AI(), sents, ["Jolene's result"])[0]["quotes"] == ["Jolene helped us"]


def test_rule_learning_uses_the_m2s_ollama_first_when_it_is_allowed(tmp_path, monkeypatch):
    from clipfarm.core import llm as L
    AV, p = _projects(tmp_path)
    p.cfg.data["clipper"]["ai_videos"]["local_projects"] = True
    asked = []

    class OpenAI:
        def complete_json(self, task, *a, **k):
            asked.append("openai")
            return _Res({"rules": []})

    class Eng:
        def router_for_now(self):
            return type("R", (), {"cfg": {"ollama_remote": {"base_url": "http://m2:11435"}}})(), "M2"
    p.engine = Eng()

    def m2_answers(self, task, system, prompt, schema, **kw):
        asked.append(tuple(self.chain))
        return _Res({"rules": ["Keep hooks under six words"]})
    monkeypatch.setattr(L.LLMRouter, "complete_json", m2_answers)
    assert p.learn("hooks were too long", "t", ai=OpenAI()) == ["Keep hooks under six words"]
    assert asked == [("ollama_remote",)]                                    # the M2 only - never this Mac's own Ollama
    p.engine.router_for_now = lambda: (None, "local (M2 allowed 17:00-08:00)")
    p.learn("captions too big", "t", ai=OpenAI())
    assert asked[-1] == "openai"                                            # by day: the AI Videos key


def test_simple_steps_use_the_cheap_model_and_the_edit_stays_on_the_main_ones(tmp_path, monkeypatch):
    from clipfarm.clipper import localai as LAI
    from clipfarm.core import llm as L
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    used = []

    def fake(self, system, prompt, schema, images=None):
        used.append(self.cfg["model"])
        return L.LLMResult({"ok": True}, "openai", self.cfg["model"], {"input_tokens": 1000, "output_tokens": 100})
    monkeypatch.setattr(L.OpenAILLM, "complete_json", fake)
    ai = LAI.LocalAI(cfg)
    ai.key = "sk-test"
    for task in ("broll_describe", "hook_text", "key_points", "story_edit", "highlight_assemble", "flow_plan", "speaker_roles", "broll_plan",
                 "multicam_director", "revise_request"):
        ai.complete_json(task, "s", "p", {"type": "object"})
    # Whole-script mapping and the final rendered-evidence audit use the accuracy model; ordinary edit decisions remain on the existing routing.
    assert used == ["gpt-4.1-nano"] * 3 + ["gpt-4.1-mini"] * 2 + ["gpt-5.4"] + ["gpt-4.1"] + ["gpt-4.1-mini"] * 3
    est = LAI.estimate_cost("gpt-4.1-mini", 3600, 10)
    assert est["llm_usd"] < LAI.estimate_cost("gpt-4.1-mini", 3600, 10, cheap_model="gpt-4.1-mini")["llm_usd"]


def test_chatgpt_plan_keeps_tokens_out_of_the_editor_and_is_preferred_over_an_api_key(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")              # sign-in is only used where the packages are installed (this Mac)
    pytest.importorskip("jwt")
    from clipfarm.clipper import chatgpt_plan as CP
    from clipfarm.clipper import localai as LAI
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    plan = CP.ChatGPTPlan(cfg)
    monkeypatch.setattr(plan, "_key", lambda: b"x" * 32)
    plan._write({"client_id": "oaiapp_test", "subject": "account", "scopes": ["chatgpt.tokens.use.direct"],
                 "credentials": {"access_token": "secret", "refresh_token": "refresh", "expires_at": 9_999_999_999}})
    assert "secret" not in plan.auth_file.read_text()
    assert plan.status()["sharing"] and plan.status().get("email") is None

    monkeypatch.setattr(LAI.ChatGPTPlan, "connected", lambda self: True)
    seen = {}
    def plan_response(self, *args, **kwargs):
        seen.update(kwargs)
        return {"ok": True}, "gpt-plan"
    monkeypatch.setattr(LAI.ChatGPTPlan, "complete_json", plan_response)
    ai = LAI.LocalAI(cfg)
    ai.key = "should-not-be-used"
    assert ai.complete_json("story_edit", "s", "p", {"type": "object"}).provider == "chatgpt_plan"
    ai.complete_json("flow_plan", "s", "p", {"type": "object"})
    assert seen["reasoning_effort"] == "xhigh"


def test_chatgpt_plan_uses_a_valid_uuid_host_and_rotates_the_unusable_pre_signin_value(tmp_path):
    import json
    import uuid
    from clipfarm.clipper import chatgpt_plan as CP
    from clipfarm.core.config import load_config

    cfg = load_config()
    cfg.root = tmp_path
    plan = CP.ChatGPTPlan(cfg)
    value = plan._host_id()
    assert value.startswith("urn:uuid:")
    assert str(uuid.UUID(value.removeprefix("urn:uuid:"))) == value.removeprefix("urn:uuid:")

    plan.host_file.write_text(json.dumps({"version": 1, "id": "urn:uuid:12345678-1234-4a-8b-123456789abc"}))
    rotated = plan._host_id()
    assert rotated != "urn:uuid:12345678-1234-4a-8b-123456789abc"
    assert list(plan.dir.glob("chatgpt-host.invalid-*.json"))


def test_chatgpt_plan_decodes_byte_sse_lines_before_parsing_them():
    from clipfarm.clipper.chatgpt_plan import _sse_text

    assert _sse_text(b"event: response.completed") == "event: response.completed"
    assert _sse_text("data: {}") == "data: {}"


def test_parakeet_is_the_default_transcriber_and_a_mac_without_it_uses_openai(tmp_path, monkeypatch):
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper import parakeet as PK
    AV, p = _projects(tmp_path)
    folder = tmp_path / "f"
    folder.mkdir()
    proj = p.create_local(str(folder), "", "", {}, {"clips": 0})
    assert p.store.get(proj["job"])["options"]["transcriber"] == "parakeet"
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    r.cfg = p.cfg
    monkeypatch.setattr(PK, "binary", lambda cfg=None: None)
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: [{"text": "hi.", "start": 0.0, "end": 0.3}])
    assert r._words_of({"path": "x.mov", "dur": 3.0}, None, "parakeet")[0]["text"] == "hi."


def test_a_timecode_final_cut_cannot_read_is_not_put_in_the_project():
    from clipfarm.clipper import localproj as LP
    sony = {"streams": [{"codec_type": "video", "codec_tag_string": "hvc1"}, {"codec_type": "data", "codec_tag_string": "rtmd",
                                                                                "tags": {"timecode": "03:51:50:08"}}]}
    assert LP.timecode_s(sony, Fraction(24000, 1001)) == 0.0 and not LP.timecode_df(sony)      # rtmd: Final Cut starts these at 0


def test_a_camera_that_started_later_is_placed_by_a_gap_because_an_angle_is_serial():
    """Final Cut: an mc-angle holds clips 'ordered serially in time', so its first clip starts at the angle's start whatever offset says -
    Camera 2 (recording from 7.38 s) played 7 s too early. The leading time must be an explicit <gap>."""
    import re

    from clipfarm.clipper import fcpxml as FX
    rate = Fraction(24000, 1001)
    mk = lambda path, t0, dur: {"path": path, "t0": t0, "dur": dur, "w": 3840, "h": 2160, "fps": 23.976023976023978, "tc": 0.0}   # noqa: E731
    group = {"angles": [{"id": "Camera 1", "files": [mk("/x/a.mp4", 0.0, 100.0)]},
                        {"id": "Camera 2", "files": [mk("/x/b1.mp4", 7.3843, 40.0), mk("/x/b2.mp4", 60.0, 30.0)]}],
             "recorders": [{"path": "/x/rec.wav", "t0": 2.5, "dur": 95.0}], "dur": 100.0}
    xml = FX.build_multicam("T", 1920, 1080, rate, group, [{"kind": "mc", "out": 0.0, "dur": 5.0, "t": 10.0, "video": "Camera 2", "audio": "Recorder"}])
    angles = dict(re.findall(r'<mc-angle name="([^"]+)"[^>]*>(.*?)</mc-angle>', xml, re.S))
    assert "<gap" not in angles["Camera 1"]                                             # it starts at 0: nothing to wait for
    cam2 = angles["Camera 2"]
    assert cam2.count("<gap") == 2 and cam2.index("<gap") < cam2.index("<asset-clip")  # a gap before the first file, one between the two
    assert re.search(r'<gap [^>]*offset="0s"[^>]*duration="1001/24000s"', cam2) is None   # (and it is the right length:)
    first_gap = re.search(r'<gap [^>]*duration="([^"]+)"', cam2).group(1)
    num, den = (first_gap.rstrip("s").split("/") + ["1"])[:2]
    assert abs(int(num) / int(den) - 7.3843) < 0.03                                       # 7.38 s, to the frame
    assert "<gap" in angles["Recorder"]


def test_script_words_are_names_only_if_the_context_says_so():
    from clipfarm.clipper import flow as FL
    interview = ("I was in design school. My mom was the director at the Harbor Clinic and Jolene helped us. The Harbor Clinic was a blessing. It was an unexpected thing, "
                 "a young mom, parenting is hard. Then Hannah Halden called.")
    outline = ["Unexpected pregnancy: 22, design school", "Jolene helped them find the clinic", "Parenting classes offered benefits",
               "Young couple, Lucia, moved to Dubuque", "Harbor enters the story: Hannah Halden, Little Steps"]
    must = FL.requirements(outline, interview)
    lit = {n: [x for x, l in v if l] for n, v in must.items()}
    assert lit[1] == ["22"]                              # "Unexpected" only starts the label: the guest says it in lower case
    assert "Jolene" in lit[2]                            # a name at the start of a step: the guest writes it with a capital mid-sentence
    assert "Parenting" not in lit[3] and "Young" not in lit[4] and "Lucia" in lit[4]
    assert "Harbor" in lit[5] and "Hannah Halden" in lit[5]
    # no interview text: the rest of the script has to vouch for a capitalised first word
    solo = FL.requirements(["Unexpected pregnancy: 22", "Harbor enters the story: Hannah, and later Harbor helps"])
    assert [x for x, l in solo[1] if l] == ["22"] and "Harbor" in [x for x, l in solo[2] if l]


def test_script_numbers_match_whole_words_only():
    from clipfarm.clipper import flow as FL
    for num, text in (("2", "we run a network of clinics"), ("1", "someone told me"), ("10", "I often came back"), ("6", "sixteen weeks")):
        assert not FL.heard(num, text), (num, text)
    assert FL.heard("22", "I was twenty two years old") and FL.heard("2", "there were two of us") and FL.heard("2019", "in twenty nineteen")


def test_a_range_that_overlaps_another_step_keeps_its_own_sentences():
    from clipfarm.clipper import flow as FL
    sents = [M.Sentence(i, i * 3.0, i * 3.0 + 2.5, f"sentence {i} about something specific and different number {i}.") for i in range(10)]

    class AI:
        def complete_json(self, task, *a, **k):
            if task == "flow_plan":
                return _Res({"title": "t", "summary": "s", "steps": [{"n": 1, "covered": True, "ranges": [{"first": 0, "last": 3}]},
                                                                      {"n": 2, "covered": True, "ranges": [{"first": 3, "last": 6}]},
                                                                      {"n": 3, "covered": True, "ranges": [{"first": 7, "last": 9}]}]})
            return _Res({"passages": [], "title": "", "summary": ""})
    items, _t, _s, rep = FL.plan(AI(), sents, ["one", "two", "three"], "notes", 200.0)
    assert [(i["step"], i["a"], i["b"]) for i in items] == [(1, 0, 3), (2, 4, 6), (3, 7, 9)] and all(r["covered"] for r in rep)


def test_the_final_check_needs_quotes_that_are_really_in_the_finished_video():
    from clipfarm.clipper import flow as FL

    class AI:
        def complete_json(self, *a, **k):
            return _Res({"steps": [{"n": 1, "missing": [], "quotes": ["she wanted to be a nurse since she was six"]},
                                   {"n": 2, "missing": [], "quotes": ["we fit every single foot by hand"]}]})
    sents = [M.Sentence(0, 0, 3, "We fit every single foot by hand here, every day.")]
    rep = FL.final_check(AI(), sents, ["Her dream of design", "How the shop works"])
    assert rep[0]["missing"] and not rep[0]["quotes"]                                 # an invented quote is no proof
    assert rep[1]["missing"] == [] and rep[1]["quotes"] == ["we fit every single foot by hand"]


def test_the_final_check_ignores_details_the_script_never_wrote():
    from clipfarm.clipper import flow as FL

    class AI:
        def complete_json(self, *a, **k):
            return _Res({"steps": [{"n": 1, "missing": ["how old we both were", "we moved to Dubio"], "quotes": ["we moved here from Hawaii last year"]}]})
    sents = [M.Sentence(0, 0, 3, "So we moved here from Hawaii last year with Lucia.")]
    rep = FL.final_check(AI(), sents, ["Couple with daughter, Lucia, moved from Hawaii to be closer to family."])
    assert "how old we both were" not in rep[0]["missing"] and "we moved to Dubio" not in rep[0]["missing"]    # the reviewer's inventions are dropped
    assert "Dubuque" not in " ".join(rep[0]["missing"])
    rep = FL.final_check(AI(), sents, ["Couple moved from Hawaii to Dubuque."])
    assert "Dubuque" in rep[0]["missing"]                                          # a name the script wrote must really be heard


def test_a_number_glued_to_the_word_before_it_is_split():
    from clipfarm.clipper import flow as FL
    from clipfarm.clipper.parakeet import split_glued_numbers
    w = [{"text": "I", "start": 0.0, "end": 0.1}, {"text": "was22", "start": 0.1, "end": 0.9}, {"text": "years", "start": 0.9, "end": 1.2},
         {"text": "it2019?", "start": 2.0, "end": 3.0}, {"text": "COVID19", "start": 3.0, "end": 3.5}, {"text": "4K", "start": 3.5, "end": 3.8}]
    got = split_glued_numbers(w)
    assert [x["text"] for x in got] == ["I", "was", "22", "years", "it", "2019?", "COVID19", "4K"]
    assert got[1]["end"] == got[2]["start"] and got[1]["start"] == 0.1 and got[2]["end"] == 0.9          # the time is shared, none lost
    assert FL.heard("22", " ".join(x["text"] for x in got))


def test_a_name_the_guest_never_says_is_not_in_the_footage_not_missing():
    from clipfarm.clipper import flow as FL

    class AI:
        def complete_json(self, *a, **k):
            raise RuntimeError("offline")
    sents = [M.Sentence(0, 0, 3, "I was pregnant at school and scared."), M.Sentence(1, 4, 6, "Hannah Halden is my mom and a director.")]
    items = [{"kind": "talk", "a": 0, "b": 0, "step": 1}, {"kind": "talk", "a": 1, "b": 1, "step": 2}]
    rep = FL.recheck(AI(), sents, items, ["The start: Alex was pregnant at school", "Her mom Hannah Halden helped"])
    assert rep[0]["not_in_footage"] == ["Alex"] and rep[0]["missing"] == [] and rep[1]["missing"] == []
    # one stray capital in chatter does not make a topic label a name
    must = FL.requirements(["Practical help: baby bucks", "x", "y"], "Right Practical Well I overthink the questions")
    assert "Practical" not in [w for w, lit in must[1] if lit]


def test_a_chatgpt_plan_answer_that_ends_early_is_retried(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    from clipfarm.clipper import chatgpt_plan as CP
    from clipfarm.clipper import localai as LAI
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    monkeypatch.setattr(LAI.time, "sleep", lambda s: None)
    monkeypatch.setattr(CP.ChatGPTPlan, "connected", lambda self: True)
    calls = []

    def flaky(self, system, prompt, schema, model=None, images=None, reasoning_effort=None):
        calls.append(1)
        if len(calls) < 3:
            raise CP.ChatGPTPlanError("ChatGPT-plan response ended before completion")
        return {"ok": True}, "gpt-5.4"
    monkeypatch.setattr(CP.ChatGPTPlan, "complete_json", flaky)
    ai = LAI.LocalAI(cfg)
    assert ai.complete_json("flow_final", "s", "p", {"type": "object"}).data == {"ok": True} and len(calls) == 3
    monkeypatch.setattr(CP.ChatGPTPlan, "complete_json", lambda self, *a, **k: (_ for _ in ()).throw(CP.ChatGPTPlanError("sign in again")))
    with pytest.raises(LAI.LLMUnavailable):
        ai.complete_json("flow_final", "s", "p", {"type": "object"})             # a sign-in problem is not retried


def test_parakeet_transcribes_a_short_file_whole_and_a_long_one_in_wide_chunks(tmp_path, monkeypatch):
    import subprocess

    from clipfarm.clipper import parakeet as PK
    from clipfarm.core import ffmpeg
    wav = tmp_path / "a.wav"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(wav)], check=True)
    exe = tmp_path / "parakeet-mlx"
    exe.write_text("")
    monkeypatch.setattr(PK, "binary", lambda cfg=None: exe)
    seen = []

    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "--chunk-duration" not in cmd and "--output-format" not in cmd:
            return real_run(cmd, **kw)                                                  # ffprobe etc. run for real
        seen.append(cmd[cmd.index("--chunk-duration") + 1:])
        out = Path(cmd[cmd.index("--output-dir") + 1]) / "audio.json"
        out.write_text(json.dumps({"sentences": [{"tokens": [{"text": " hi", "start": 0.0, "end": 0.3}]}]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    from pathlib import Path
    monkeypatch.setattr(PK.subprocess, "run", fake_run)
    assert PK.transcribe(wav)[0]["text"] == "hi" and seen == [["0"]]                      # short: in one piece
    monkeypatch.setattr(PK, "WHOLE_FILE_S", 1.0)
    PK.transcribe(wav)
    assert seen[-1] == ["300", "--overlap-duration", "30"]                                  # long: wide chunks, wide overlap


def test_timecodes_final_cut_cannot_read_are_cleared_from_a_saved_multicam_group(monkeypatch):
    from clipfarm.clipper import localproj as LP
    info = {"streams": [{"codec_type": "video", "codec_tag_string": "hvc1", "r_frame_rate": "24000/1001", "avg_frame_rate": "24000/1001"},
                        {"codec_type": "data", "codec_tag_string": "rtmd", "tags": {"timecode": "03:51:50:08"}}]}
    monkeypatch.setattr(LP.ffmpeg, "probe", lambda p: info)
    group = {"angles": [{"id": "Camera 1", "files": [{"path": "/x/C0501.MP4", "fps": 23.976, "tc": 13924.24, "tc_df": False}]}]}
    LP.refresh_timecodes([], group)
    assert group["angles"][0]["files"][0]["tc"] == 0.0                    # an old multicam.json's rtmd-based start is replaced


def test_one_long_broll_shot_supplies_many_cutaways():
    from clipfarm.clipper import broll as BR
    nine_min = [{"start": 100.0, "end": 640.0, "what": "a slow pan over the lake shore", "quality": 8.0}]
    parts = BR.split_long(nine_min)
    assert 40 >= len(parts) >= 30 and parts[0]["start"] == 100.0 and abs(parts[-1]["end"] - 640.0) < 0.1
    assert all(p["end"] - p["start"] <= 20 and "part" in p["what"] for p in parts)               # each part is its own, different stretch
    assert all(a["end"] <= b["start"] + 0.01 for a, b in zip(parts, parts[1:]))                   # they don't overlap
    short = [{"start": 0.0, "end": 9.0, "what": "a dog", "quality": 7.0}, {"start": 5.0, "end": 6.0, "what": "photo", "quality": 7.0, "photo": True}]
    assert BR.split_long(short) == short                                                          # short shots and photos are left alone
    assert BR.split_long(parts) == parts                                                          # already split: not split again
    assert BR.sheets_for(30) == 2 and BR.sheets_for(540) == 14 and BR.sheets_for(7200) == 16        # long files are looked at more closely


def test_a_camera_with_almost_no_sound_still_lines_up_with_the_loud_one():
    import numpy as np
    from clipfarm.clipper import audiosync as AS
    rng = np.random.default_rng(3)
    t = np.arange(AS.SR * 60) / AS.SR
    speech = rng.standard_normal(len(t)) * (np.abs(np.sin(2 * np.pi * 0.7 * t)) * (np.sin(2 * np.pi * 0.11 * t) > -0.3)) * 0.2
    loud, quiet = AS.envelope(speech), AS.envelope(speech * 1.5e-3)               # the same room, -57 dB lower (a microphone that was barely plugged in)
    assert np.corrcoef(loud, quiet)[0, 1] > 0.9


def test_a_short_cut_is_topped_up_toward_the_length_that_was_asked_for(monkeypatch):
    from clipfarm.clipper import highlights as HL
    from clipfarm.clipper import localproj as LP
    sents = [M.Sentence(i, i * 10.0, i * 10.0 + 9.0, f"sentence {i}.") for i in range(30)]
    cands = [{"first": i, "last": i, "start": i * 10.0, "end": i * 10.0 + 9.0, "score": 10 - (i % 5), "what": "x", "id": i} for i in range(30)]
    monkeypatch.setattr(HL, "candidates", lambda *a, **k: cands)
    runner = LP.LocalRunner.__new__(LP.LocalRunner)
    monkeypatch.setattr(LP.LocalRunner, "intervals_of", lambda self, plan, sents_, words, *a, **k: [(sents[it["a"]].start, sents[it["b"]].end) for it in plan["items"]
                                                                                                      for _ in [0]])
    items = [{"kind": "talk", "a": 2, "b": 2}, {"kind": "talk", "a": 20, "b": 20}]
    out = runner.top_up(None, sents, [], items, 120.0, 0.5, [], "", "story", lambda *a: None)
    got = sum(sents[i["b"]].end - sents[i["a"]].start for i in out)
    assert got >= 0.9 * 120 and len({(i["a"], i["b"]) for i in out}) == len(out)               # no repeats
    assert [i["a"] for i in out] == sorted(i["a"] for i in out)                                  # still in the order they happened


def test_sound_runs_on_without_a_break_when_the_angle_changes():
    from fractions import Fraction
    from clipfarm.clipper import localproj as LP
    rate = Fraction(24000, 1001)

    def part(d_orig, a_ss, v_path, v_ss, t, cut=False):
        n = round(d_orig * float(rate))
        return {"d": float(Fraction(n) / rate), "n": n, "a": {"path": "A.mp4", "ss": a_ss, "pre": 0.0},
                "v": {"path": v_path, "ss": v_ss, "fill": cut, **({"cut": True} if cut else {})},
                "item": {"kind": "mc", "t": t, "video": v_path, "audio": "Camera 1"}}
    # a 10.552 s piece on camera 1 (rounds to 253 frames = 10.5522 s) then the SAME speech continues on camera 2 at the original 10.552 s later
    first = part(10.552, 100.0, "A.mp4", 100.0, 100.0)
    second = part(5.0, 110.552, "B.mp4", 110.7, 110.552)
    third = part(3.0, 115.552, "Broll.mp4", 7.0, 115.552, cut=True)
    first["d"] = float(Fraction(252) / rate)                                   # quantize shortened it by a frame
    out = LP.seal_joins([first, second, third])
    assert abs(out[1]["a"]["ss"] - (first["a"]["ss"] + first["d"])) < 1e-9                      # the sound starts exactly where it stopped
    assert abs((out[1]["v"]["ss"] - 110.7) - (out[1]["a"]["ss"] - 110.552)) < 1e-9               # the picture moved with it: lip-sync kept
    assert abs(out[1]["item"]["t"] - out[1]["a"]["ss"]) < 1e-9                                   # Final Cut's clip starts at the same place
    assert abs(out[2]["a"]["ss"] - (out[1]["a"]["ss"] + out[1]["d"])) < 1e-9 and out[2]["v"]["ss"] == 7.0   # B-roll keeps its own place
    real_jump = [part(4.0, 0.0, "A.mp4", 0.0, 0.0), part(4.0, 30.0, "A.mp4", 30.0, 30.0)]
    assert LP.seal_joins([dict(p) for p in real_jump])[1]["a"]["ss"] == 30.0                    # a real cut (skipped speech) is left alone


def test_final_cut_gets_only_the_broll_that_was_played_and_inside_its_file():
    from clipfarm.clipper import localproj as LP
    files = [{"path": "/b/C4693.MP4", "w": 3840, "h": 2160, "fps": 23.976, "dur": 550.55, "tc": 0.0}]
    parts = [{"d": 3.5, "out": 70.0, "v": {"path": "/b/C4693.MP4", "ss": 100.0, "cut": True}},
             {"d": 4.0, "out": 80.0, "v": {"path": "/b/C4693.MP4", "ss": 200.0, "cut": True}},
             {"d": 2.0, "out": 84.0, "v": {"path": "/b/C4693.MP4", "ss": 204.0, "cut": True}},             # the same shot, cut in two by the render
             {"d": 3.5, "out": 90.0, "v": {"path": "/b/C4693.MP4", "ss": 550.0, "cut": True}},             # starts right at the end of the file
             {"d": 5.0, "out": 95.0, "v": {"path": "/b/C0080.MP4", "ss": 10.0}}]                           # a camera, not B-roll
    got = LP.cutaways_of_parts(parts, files)
    assert [(c["at"], round(c["len"], 2)) for c in got[:2]] == [(70.0, 3.5), (80.0, 6.0)]                 # the two halves are one shot again
    assert all(c["start"] + c["len"] <= 550.55 for c in got) and len(got) == 3 and got[2]["len"] < 0.6   # the one at the very end is cut back to fit


def test_um_uh_and_stuttered_words_are_cut_out_of_the_speech():
    from clipfarm.clipper import localproj as LP
    def w(text, a, b): return {"text": text, "start": a, "end": b}
    words = [w("So", 0.0, 0.3), w("it", 0.35, 0.5), w("was", 0.55, 0.9), w("um,", 1.0, 1.5), w("really", 1.6, 2.1), w("hard", 2.15, 2.6),
             w("and", 2.7, 2.9), w("I", 3.0, 3.1), w("I", 3.15, 3.25), w("went", 3.3, 3.7), w("home", 3.75, 4.2), w("very", 4.3, 4.5), w("very", 4.55, 4.8),
             w("tired.", 4.85, 5.4)]
    got = LP.cut_fillers([(0.0, 5.5)], words)
    assert len(got) == 3 and abs(got[0][1] - 1.025) < 1e-6 and abs(got[1][0] - 1.475) < 1e-6          # the "um" is gone (a hair of margin each side)
    assert got[1][1] < 3.1 and got[2][0] > 3.1                                                      # the first "I" of "I I" is gone, the second stays
    assert got[2][1] == 5.5                                                                         # "very very" is a real emphasis: untouched
    at_start = LP.cut_fillers([(1.0, 5.5)], words)                                                # a filler at the very start just trims the start
    assert at_start[0][0] > 1.4
    sliver = LP.cut_fillers([(0.7, 3.0)], words)                                                  # 0.3 s of "was" would be left alone before the "um": keep it
    assert sliver[0] == (0.7, 1.025) or sliver == [(0.7, 3.0)]
    soft = LP.cut_fillers([(0.0, 5.5)], words, policy="keep_soft")                                # keep_soft: this "um" is long (0.5 s), so it still goes...
    assert soft == got
    assert LP.cut_fillers([(0.0, 5.5)], words, policy="keep") == [(0.0, 5.5)]                      # ...and "keep" leaves everything


def test_jump_cuts_on_one_camera_get_an_alternating_punch_in():
    from clipfarm.clipper import localproj as LP
    def piece(path, ss, d, cut=False):
        return {"d": d, "a": {"path": "A.mp4", "ss": ss, "pre": 0.0}, "v": {"path": path, "ss": ss, **({"cut": True} if cut else {})},
                "item": {"kind": "mc", "t": ss, "video": "Camera 1", "audio": "Camera 1"}}
    parts = [piece("A.mp4", 0.0, 5.0), piece("A.mp4", 20.0, 5.0), piece("A.mp4", 40.0, 5.0), piece("A.mp4", 60.0, 5.0),
             piece("B.mp4", 80.0, 5.0), piece("A.mp4", 100.0, 5.0)]
    out = LP.punch_ins(parts, 1.12)
    assert [bool(p.get("zoom")) for p in out] == [False, True, False, True, False, False]          # every second jump cut; a camera change needs none
    assert out[1]["item"]["zoom"] == 1.12                                                           # Final Cut gets the same scale
    contiguous = [piece("A.mp4", 0.0, 5.0), piece("A.mp4", 5.0, 5.0)]
    assert not any(p.get("zoom") for p in LP.punch_ins(contiguous))                                  # a continuous take is not a cut at all
