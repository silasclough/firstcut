"""Learning an editing style from a video (stylelearn.py): the measures against synthetic videos with a KNOWN cut rhythm, captions through a fake text
reader, combining videos, clamping into the style's ranges, the points (fake AI and the no-AI template), and that nothing but numbers is kept."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipfarm.clipper import stylefetch, stylelearn as SL
from clipfarm.clipper import styles_profile as SPF
from clipfarm.core import ffmpeg

COLOURS = ["black", "white"] * 8


def _cut_video(path: Path, every: float, secs: float = 12.0, size: str = "320x568", audio: bool = True) -> Path:
    """A video that changes picture every `every` seconds (solid colours), with a steady tone so there is sound."""
    n = max(1, int(round(secs / every)))
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error"]
    for k in range(n):
        cmd += ["-f", "lavfi", "-i", f"color=c={COLOURS[k % len(COLOURS)]}:s={size}:r=20:d={every}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:d={secs}"]
    cmd += ["-filter_complex", "".join(f"[{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p[v]", "-map", "[v]"]
    if audio:
        cmd += ["-map", f"{n}:a", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", str(path)], check=True, timeout=120)
    return path


def _m(**kw) -> dict:
    base = {k: None for k in SL.NUMBERS}
    base.update({k: None for k in SL.WORDS})
    base.update({"duration_s": 30.0, "aspect": "9:16", "faces_measured": False})
    base.update(kw)
    return base


# ----------------------------------------------------------------------------------------------------------------- the measures
@pytest.mark.parametrize("every,low,high", [(1.0, 25, 31), (2.0, 11, 14), (4.0, 5, 8)])
def test_a_known_cut_rhythm_is_measured(tmp_path, every, low, high):
    v = _cut_video(tmp_path / "v.mp4", every, 12.0)
    m = SL.measure(v, ocr_read=lambda imgs: None)
    assert low <= m["cuts_per_30s"] <= high                                              # a cut every 2 s is about 12.5 cuts in 30 s
    assert abs(m["shot_median_s"] - every) < 0.35 and m["aspect"] == "9:16" and abs(m["duration_s"] - 12.0) < 0.2
    assert m["hook_cuts"] == (2 if every == 1.0 else 1 if every == 2.0 else 0)           # the cuts inside the first 3 seconds


def test_no_text_reader_means_captions_are_not_measured_never_guessed(tmp_path):
    m = SL.measure(_cut_video(tmp_path / "v.mp4", 2.0), ocr_read=lambda imgs: None)
    assert m["caption_share"] is None and m["caption_words_median"] is None and m["caption_colour"] is None
    agg = SL.aggregate([m])
    assert "captions" in SL.evidence(agg)[-1] and "captions" not in SPF.flat_from_nested(SL.to_style(agg)[0])
    assert "captions" not in SL.to_style(agg)[0]


def test_captions_are_read_with_their_words_position_case_and_colour(tmp_path):
    v = _cut_video(tmp_path / "v.mp4", 12.0, 12.0)                                       # one solid black picture
    lines = [{"t": "WAIT FOR THIS", "x": 0.2, "y": 0.68, "w": 0.6, "h": 0.06, "c": 0.9}]
    m = SL.measure(v, ocr_read=lambda imgs: [list(lines) for _ in imgs])
    assert m["caption_share"] == 1.0 and m["caption_words_median"] == 3 and m["caption_case"] == "upper"
    assert abs(m["caption_y"] - 0.71) < 0.02 and abs(m["caption_height"] - 0.06) < 0.01
    assert m["caption_colour"] in ("black", "white", "grey")                             # plain words on a plain picture: no highlight colour
    # a promo bar, a title near the top and a watermark at the very bottom are not captions
    other = [{"t": "SUBSCRIBE @someone", "x": 0.1, "y": 0.75, "w": 0.5, "h": 0.05, "c": 0.9}, {"t": "TOP TITLE", "x": 0.1, "y": 0.05, "w": 0.5, "h": 0.05, "c": 0.9},
             {"t": "logo", "x": 0.1, "y": 0.95, "w": 0.1, "h": 0.03, "c": 0.9}]
    m2 = SL.measure(v, ocr_read=lambda imgs: [list(other) for _ in imgs])
    assert m2["caption_share"] == 0.0 and m2["caption_words_median"] is None


def test_the_colour_of_words_and_a_yellow_highlight():
    img = np.zeros((100, 100, 3), np.uint8)
    img[40:60, 10:90] = (255, 255, 0)
    assert SL.line_colour(img, (0.1, 0.4, 0.8, 0.2)) == "yellow"
    img[40:60, 10:90] = (255, 255, 255)
    assert SL.line_colour(img, (0.1, 0.4, 0.8, 0.2)) == "white"
    assert SL.line_colour(img, (0.1, 0.4, 0.01, 0.01)) is None


def test_a_music_bed_under_talking_is_told_from_silence_between_words(tmp_path):
    segs = [[0.0, 3.0], [4.0, 7.0], [8.0, 11.0]]
    tone = _cut_video(tmp_path / "a.mp4", 12.0, 12.0)                                    # a steady tone the whole way: gaps are not quiet
    a = SL.audio_levels(tone, 12.0, segs)
    assert a["music"] == "under" and a["speech_db"] is not None
    silent = tmp_path / "s.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x112:r=10:d=12", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
                    "-t", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", str(silent)], check=True)
    assert SL.audio_levels(silent, 12.0, segs)["music"] == "none"


def test_speech_pauses_are_the_gaps_between_the_talking():
    s = SL.speech_stats([[0, 2], [2.5, 5], [5.6, 8], [8.4, 10]], 10.0)
    assert s["speech_ratio"] == pytest.approx(0.85, abs=0.02) and s["pause_median_s"] == pytest.approx(0.5, abs=0.11)
    assert SL.speech_stats([], 10.0)["pause_median_s"] is None


def test_a_video_too_short_to_learn_from_is_refused_in_words(tmp_path):
    with pytest.raises(stylefetch.FetchError, match="too short"):
        SL.measure(_cut_video(tmp_path / "v.mp4", 1.0, 1.0), ocr_read=lambda i: None)


# ----------------------------------------------------------------------------------------------------------------- combining and writing
def test_several_videos_give_median_spread_and_what_repeats():
    ms = [_m(cuts_per_30s=10.0, shot_median_s=2.8, caption_share=0.9, music="under"), _m(cuts_per_30s=12.0, shot_median_s=2.4, caption_share=0.8, music="under"),
          _m(cuts_per_30s=30.0, shot_median_s=0.9, caption_share=0.9, music="none")]
    agg = SL.aggregate(ms)
    n = agg["numbers"]
    assert agg["videos"] == 3 and n["cuts_per_30s"]["median"] == 12.0 and n["cuts_per_30s"]["lo"] == 10.0 and n["cuts_per_30s"]["hi"] == 30.0
    assert n["caption_share"]["consistent"] is True and n["cuts_per_30s"]["consistent"] is False        # repeats against varies
    assert agg["words"]["music"] == {"value": "under", "share": 0.67, "n": 3}


def test_the_numbers_become_a_valid_style_even_when_they_are_absurd():
    agg = SL.aggregate([_m(cuts_per_30s=80.0, shot_p90_s=500.0, shot_p25_s=0.1, shot_p75_s=900.0, pause_median_s=5.0, no_face_share=3.0, caption_share=1.0,
                           caption_words_median=40, caption_words_p25=30, caption_words_p75=50, caption_y=0.99, caption_case="upper", caption_colour="yellow",
                           duration_s=5000.0, music="under")])
    p, notes = SL.to_style(agg)
    base = SPF.resolve("longform_talk")
    new, problems = SPF.apply_partial(base, p)
    assert problems == [] and new["pace"]["name"] == "fast"
    assert new["pace"]["max_hold_s"] <= 30 and new["pace"]["gap_shrink_to_s"] <= 0.6 and 0.5 <= new["broll"]["shot_s"][0] <= new["broll"]["shot_s"][1]
    assert 0 <= new["broll"]["share"] <= 1 and new["captions"]["style"] == "phrase" and new["captions"]["words_per_group"][1] <= 12
    assert new["captions"]["highlight"] == "one_colour" and 0 <= new["captions"]["band_y"][0] < new["captions"]["band_y"][1] <= 0.95 and new["music"]["policy"] == "throughout"


def test_a_short_vertical_video_sets_its_length_a_horizontal_one_does_not():
    short = SL.to_style(SL.aggregate([_m(duration_s=28.0, aspect="9:16", cuts_per_30s=9.0)]))[0]
    assert short["length"]["target_s"] == 28 and short["pace"]["name"] == "tight"
    wide = SL.to_style(SL.aggregate([_m(duration_s=170.0, aspect="16:9", cuts_per_30s=2.0)]))[0]
    assert "length" not in wide and wide["pace"]["name"] == "slow"


def test_no_music_and_no_captions_are_settings_too():
    p = SL.to_style(SL.aggregate([_m(caption_share=0.0, music="none", cuts_per_30s=5.0)]))[0]
    assert p["captions"] == {"style": "none"} and p["music"] == {"policy": "none"}


def test_a_learned_style_never_touches_the_hard_rules():
    allowed = set(SPF.SETTING_KEYS)
    p = SL.to_style(SL.aggregate([_m(cuts_per_30s=9.0, caption_share=0.9, caption_words_median=2, music="under", no_face_share=0.5)]))[0]
    assert set(p) <= {"pace", "broll", "captions", "music", "length"}                    # only editing numbers: nothing about gates, gameplay or posting
    assert set(SPF.flat_from_nested(p)) <= allowed


class FakeAI:
    def __init__(self, answer=None, boom=False):
        self.answer, self.boom, self.calls = answer, boom, []

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        self.calls.append({"task": task, "system": system, "prompt": prompt, "images": images})
        if self.boom:
            raise RuntimeError("no AI today")
        return SimpleNamespace(data=self.answer)


def test_the_points_come_from_one_cheap_call_that_may_only_state_what_was_measured():
    agg = SL.aggregate([_m(cuts_per_30s=12.0, shot_median_s=2.0, shot_p90_s=3.0, caption_share=0.9, caption_words_median=2, music="under")])
    ai = FakeAI({"points": ["Cut every 2 seconds.", "Two-word captions.", "Music under everything."]})
    pts, how = SL.write_points(ai, agg, [b"jpeg1", b"jpeg2"])
    assert how == "ai" and pts == ["Cut every 2 seconds.", "Two-word captions.", "Music under everything."]
    c = ai.calls[0]
    assert c["task"] == "style_learn" and "not measured" in c["system"].lower() and "12" in c["prompt"] and c["images"] == [b"jpeg1", b"jpeg2"]
    assert "style_learn" in __import__("clipfarm.clipper.localai", fromlist=["x"]).CHEAP_TASKS


@pytest.mark.parametrize("ai", [FakeAI(boom=True), FakeAI({"points": "not a list of lines"}), FakeAI({"points": []}), FakeAI(None), None])
def test_without_a_usable_answer_the_points_are_written_from_the_numbers(ai):
    agg = SL.aggregate([_m(cuts_per_30s=12.0, shot_median_s=2.0, shot_p90_s=3.0, pause_median_s=0.3, caption_share=0.9, caption_words_median=2, caption_y=0.7,
                           caption_case="upper", caption_colour="yellow", music="under", zoom_per_30s=3.0)])
    pts, how = SL.write_points(ai, agg)
    text = " ".join(pts)
    assert how in ("numbers", "ai") and 3 <= len(pts) <= 8 and all(len(p) <= 160 for p in pts)
    if how == "numbers":
        assert "tight pace" in text and "0.3" in text and "yellow" in text and "Music under" in text and "zoom" in text.lower()


def test_the_evidence_lines_name_the_numbers_and_what_was_not_measured():
    agg = SL.aggregate([_m(cuts_per_30s=14.0, shot_median_s=2.1, shot_p90_s=3.5, hook_cuts=2, speech_ratio=0.8, pause_median_s=0.3, caption_share=0.9,
                           caption_words_median=3, caption_y=0.7, caption_case="upper", caption_colour="yellow")])
    ev = SL.evidence(agg)
    assert ev[0].startswith("14 cuts in 30 s; the middle shot lasts 2.1 s") and any("2 cuts in the first 3 seconds" in x for x in ev)
    assert any("Captions on screen 90%" in x and "yellow highlight" in x for x in ev) and ev[-1].startswith("Not measured:") and "graphics" in ev[-1]


# ----------------------------------------------------------------------------------------------------------------- one whole run
def test_learn_measures_then_deletes_every_download_and_keeps_only_numbers(tmp_path):
    made = {}

    def fetch(source, work, endpoint="", progress=None):
        work.mkdir(parents=True, exist_ok=True)
        p = _cut_video(work / "v0.mp4", 2.0)
        made["path"] = p
        return [{"path": p, "title": "A video", "url": "https://www.youtube.com/watch?v=x", "kind": "youtube_video", "owned": True}]
    work = tmp_path / "w"
    steps = []
    res = SL.learn("https://www.youtube.com/watch?v=x", fetch=fetch, ocr_read=lambda imgs: None, work=work, progress=steps.append, frames_dir=tmp_path / "frames")
    assert not work.exists() and not made["path"].exists()                               # the download and the small copies are gone
    assert steps and len(res["sources"]) == 1 and 1 <= len(res["frames"]) <= SL.KEEP_FRAMES and all(f.exists() and f.stat().st_size < 30000 for f in res["frames"])
    blob = json.dumps(res["sources"])
    for word in ("transcript", "\"words\"", "\"text\"", "\"lines\""):
        assert word not in blob                                                          # numbers, a title and a link: never the video's words
    assert set(res["sources"][0]["measurements"]) <= set(SL.NUMBERS) | set(SL.WORDS) | {"width", "height", "faces_measured"}


def test_learn_cleans_up_when_it_fails_and_never_deletes_a_persons_own_file(tmp_path):
    mine = _cut_video(tmp_path / "mine.mp4", 2.0, 1.0)                                   # too short: refused
    work = tmp_path / "w"
    with pytest.raises(stylefetch.FetchError):
        SL.learn(str(mine), ocr_read=lambda i: None, work=work)
    assert mine.exists() and not work.exists()


def test_one_unreadable_video_does_not_end_a_channel(tmp_path):
    good = _cut_video(tmp_path / "good.mp4", 2.0)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")

    def fetch(source, work, endpoint="", progress=None):
        return [{"path": bad, "title": "bad", "url": "u1", "kind": "youtube_channel", "owned": False}, {"path": good, "title": "good", "url": "u2", "kind": "youtube_channel", "owned": False}]
    res = SL.learn("https://www.youtube.com/@x", fetch=fetch, ocr_read=lambda i: None, work=tmp_path / "w")
    assert [s["title"] for s in res["sources"]] == ["good"] and good.exists() and bad.exists()


def test_summarize_ties_it_together(tmp_path):
    srcs = [{"url": "a", "title": "a", "measurements": _m(cuts_per_30s=12.0, shot_median_s=2.0, shot_p90_s=3.0, caption_share=0.9, caption_words_median=2, music="under")}]
    out = SL.summarize(srcs, FakeAI({"points": ["Cut every 2 s."]}))
    assert out["points"] == ["Cut every 2 s."] and out["how"] == "ai" and out["flat"]["pace"] == "tight" and out["flat"]["captions"] == "karaoke" and out["evidence"]
