"""WP9: B-roll motivated by the words (brollplace.py, visualnouns.py) and whose sound is in the video (brollsound.py). Offline: fake AI, real ffmpeg
for the two renders that check what the delivered audio contains."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipfarm.clipper import aivideo as AV
from clipfarm.clipper import brollplace as BP
from clipfarm.clipper import brollsound as BS
from clipfarm.clipper import visualnouns as VN
from clipfarm.clipper import moments as M
from clipfarm.clipper import styles_profile as SPF
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config
from test_aivideo import _FakeAI, _Res

FF = ffmpeg.bin_path("ffmpeg")


# ------------------------------------------------------------------------------------------------ helpers
def _words(text: str, t0: float = 0.5, step: float = 0.5) -> list[dict]:
    return [{"text": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.35, 3)} for i, w in enumerate(text.split())]


def _scene(what: str, file: str = "C0001.MP4", start: float = 0.0, **kw) -> dict:
    return {"what": what, "file": file, "start": start, "end": start + 8.0, "quality": 7, **kw}


def _noun(words, phrase: str, concrete: float = 0.9, kind: str = "object") -> dict:
    toks = phrase.split()
    norm = [w["text"].strip(",.").lower() for w in words]
    j = next(k for k in range(len(norm)) if norm[k:k + len(toks)] == toks)
    return {"sent": 0, "word_first": j, "word_last": j + len(toks) - 1, "phrase": phrase, "kind": kind, "concrete": concrete}


STYLE = SPF.resolve("longform_talk")


# ------------------------------------------------------------------------------------------------ the matcher and the placement (pure)
def test_a_cutaway_starts_a_lead_before_the_noun_and_ends_at_the_clause():
    words = _words("and then we pour the coffee, which is hot and strong")
    n = _noun(words, "coffee")
    pool = [_scene("a cup of hot coffee on a counter"), _scene("a wide street at dusk", "C0002.MP4")]
    cs, un = BP.place([n], pool, words, STYLE, 60.0, intro_free=0.0)
    assert len(cs) == 1 and not un
    c = cs[0]
    noun_start = words[n["word_first"]]["start"]
    assert c["at"] == pytest.approx(noun_start - STYLE["broll"]["lead_s"], abs=0.011)
    comma_end = words[n["word_first"]]["end"]
    lo, hi = STYLE["broll"]["shot_s"]
    assert c["at"] + c["length"] == pytest.approx(comma_end, abs=0.02) or not (lo <= comma_end - c["at"] <= hi)      # ends at the clause when that is in range
    assert lo - 1e-6 <= c["length"] <= hi + 1e-6 and c["device"] == "noun" and c["noun"] == "coffee" and c["score"] >= BP.THRESHOLD


class _TagAI:
    """visual_nouns answers: nouns and tags per sentence index."""

    def __init__(self, rows):
        self.rows, self.calls = rows, 0

    def complete_json(self, task, system, prompt, schema, **kw):
        self.calls += 1
        assert task == "visual_nouns"
        return _Res({"sentences": self.rows})


def test_protected_sentences_never_get_a_cutaway():
    words = _words("Welcome to the farm. The tractor is red. Let us pray for the harvest. The barn is old. That is the whole point of today.", step=0.6)
    sents = M.sentences_from_words(words)
    assert len(sents) == 5
    rows = [{"i": 0, "tags": ["hook"], "nouns": [{"phrase": "farm", "kind": "place", "concrete": 0.9}]},
            {"i": 1, "nouns": [{"phrase": "tractor", "kind": "object", "concrete": 1.0}]},
            {"i": 2, "tags": ["prayer"], "nouns": [{"phrase": "harvest", "kind": "object", "concrete": 0.8}]},
            {"i": 3, "tags": ["punchline"], "nouns": [{"phrase": "barn", "kind": "place", "concrete": 0.9}]},
            {"i": 4, "tags": ["key_statement"]}]
    pool = [_scene("a red tractor in a field"), _scene("an old barn at sunrise", "C0002.MP4"), _scene("wheat harvest", "C0003.MP4"), _scene("a farm gate", "C0004.MP4")]
    style = json.loads(json.dumps(STYLE))
    style["broll"]["share"] = 0.9
    out, un, info = BP.run(_TagAI(rows), {}, sents, words, [], pool, style, 40.0)
    prot = VN.protected_spans(info["tags"], sents, style["broll"]["protect"])
    assert prot and out
    for c in out:
        assert not any(c["at"] < b and c["at"] + c["length"] > a for a, b in prot), (c, prot)
    assert all("tractor" in c["what"] or "tractor" in c.get("noun", "") for c in out) or len(out) >= 1


def test_the_matcher_abstains_when_nothing_is_close_and_the_speaker_stays():
    words = _words("we poured the coffee slowly this morning in the kitchen today")
    n = _noun(words, "coffee")
    pool = [_scene("a wide street at dusk"), _scene("a mountain lake", "C0002.MP4"), _scene("people clapping in a hall", "C0003.MP4")]
    i, score, why = BP.match(n, pool, STYLE)
    assert i is None and "close" in why or "stand" in why
    cs, un = BP.place([n], pool, words, STYLE, 60.0, intro_free=0.0)
    assert cs == [] and un and un[0]["phrase"] == "coffee"


def test_a_file_is_not_reused_within_the_reuse_gap():
    words = _words("the coffee is hot and then another coffee is cold")
    n = _noun(words, "coffee")
    pool = [_scene("a cup of coffee", "A.MP4"), _scene("a cup of coffee on a table", "B.MP4")]
    i1, s1, _ = BP.match(n, pool, STYLE, recent=[], now=100.0)
    first_file = pool[i1]["file"]
    i2, s2, _ = BP.match(n, pool, STYLE, recent=[(first_file, 100.0)], now=110.0)
    assert i2 is not None and pool[i2]["file"] != first_file                         # the other file wins when this one was just used
    i3, _, _ = BP.match(n, [pool[i1]], STYLE, recent=[(first_file, 100.0)], now=300.0)
    assert i3 == 0                                                                    # 200 s later (past the 150 s gap) it is fine again


def test_portrait_and_mood_mismatches_are_penalised_not_excluded():
    n = {"phrase": "coffee", "kind": "object", "concrete": 0.9, "mood": "calm"}
    ok = _scene("a cup of coffee on a table")
    bad = _scene("a cup of coffee on a table", "C0002.MP4", mismatch=True, mood="busy")
    pool = [ok, bad, _scene("a street", "C0003.MP4"), _scene("a lake", "C0004.MP4")]
    _, s_ok, _ = BP.match(n, [ok, pool[2], pool[3]], STYLE)
    i, s_bad, _ = BP.match(n, [bad, pool[2], pool[3]], STYLE)
    assert i == 0 and s_bad < s_ok                                                    # still chosen (the only good match), scored lower


def test_total_broll_follows_the_style_share_and_the_face_is_back_within_the_limit():
    text = " ".join(f"then the coffee{i} pours slowly here and now" for i in range(12))
    words = _words(text, step=0.7)
    nouns = [_noun(words, f"coffee{i}") for i in range(12)]
    pool = [_scene(f"a cup of coffee{i}", f"C{i:04d}.MP4") for i in range(12)]
    style = json.loads(json.dumps(STYLE))
    style["broll"].update({"share": 0.25, "face_back_within_s": 4.0, "reuse_gap_s": 0})
    out_len = words[-1]["end"] + 1
    cs, _ = BP.place(nouns, pool, words, style, out_len, intro_free=0.0)
    assert cs and sum(c["length"] for c in cs) <= 0.25 * out_len + 1e-6
    assert all(c["length"] <= 4.0 + 1e-6 for c in cs)
    for a, b in zip(cs, cs[1:]):
        assert b["at"] - (a["at"] + a["length"]) >= BP.SPEAKER_BETWEEN_S - 1e-6      # the speaker is back for at least 1.5 s between shots


def test_beat_snap_never_moves_a_cutaway_onto_a_word():
    words = [{"text": "alpha", "start": 1.0, "end": 1.6}, {"text": "beta", "start": 2.0, "end": 2.6}, {"text": "coffee", "start": 3.0, "end": 3.5}]
    assert BP.snap_beat(2.85, [2.3, 2.9], words, noun_start=3.0) == 2.9               # a gap beat: fine
    assert BP.snap_beat(2.4, [2.3, 3.2], words, noun_start=3.0) == 2.4                # 2.3 is inside "beta", 3.2 is after the noun: stays
    assert BP.snap_beat(2.4, [], words) == 2.4


def test_photos_are_placed_only_where_they_match():
    words = _words("we walked past the old barn and then talked about lunch with friends for a while")
    nouns = [_noun(words, "barn")]
    pool = [_scene("an old red barn", "P1.JPG", photo="P1.JPG"), _scene("a birthday cake", "P2.JPG", photo="P2.JPG")]
    barn = {"at": words[_noun(words, "barn")["word_first"]]["start"] - 0.2, "length": 2.5, "scene": 0, "photo": "P1.JPG"}
    cake = {"at": 30.0, "length": 2.5, "scene": 1, "photo": "P2.JPG"}
    assert BP.photo_ok(barn, nouns, words, pool, STYLE) is True
    assert BP.photo_ok(cake, nouns, words, pool, STYLE) is False
    style = json.loads(json.dumps(STYLE))
    style["broll"]["photos"] = "all"
    assert BP.photo_ok(cake, nouns, words, pool, style) is True


def test_unmatched_number_nouns_go_to_graphics_before_stock():
    words = _words("we served 400 meals and baked fresh bread every single morning here at the shop")
    num = {"sent": 0, "word_first": 2, "word_last": 2, "phrase": "400 meals", "kind": "number", "concrete": 0.9}
    bread = _noun(words, "bread")
    pool = [_scene("a mountain lake")]
    cs, un = BP.place([num, bread], pool, words, STYLE, 60.0, intro_free=0.0)
    kinds = {u["phrase"]: u["kind"] for u in un}
    assert cs == [] and kinds == {"400 meals": "number", "bread": "object"}
    assert BP.stock_queries(un) == ["bread"]                                          # stock only for a concrete thing; the number is a graphic's job


def test_visual_nouns_get_word_numbers_from_code_never_from_the_model_and_are_cached():
    words = _words("welcome everyone we are baking bread in the old oven today")
    sents = M.sentences_from_words(words)
    rows = [{"i": 0, "nouns": [{"phrase": "old oven", "kind": "object", "concrete": 1.0}, {"phrase": "unicorn", "kind": "object", "concrete": 1.0}],
             "tags": ["hook", "made_up_tag"]},
            {"i": 99, "nouns": [{"phrase": "bread", "kind": "object", "concrete": 1.0}]}]
    ai, cache = _TagAI(rows), {}
    got = VN.extract(ai, sents, words, cache)
    assert [n["phrase"] for n in got["nouns"]] == ["old oven"]                         # a phrase that is not said, and a sentence the code never offered, are dropped
    n = got["nouns"][0]
    assert words[n["word_first"]]["text"] == "old" and words[n["word_last"]]["text"] == "oven"
    assert got["tags"] == {0: ["hook"]}
    again = VN.extract(ai, sents, words, cache)
    assert ai.calls == 1 and again["nouns"] == got["nouns"]                            # the second ask is answered from the cache


# ------------------------------------------------------------------------------------------------ whose sound is in the video (pure)
def _file(name: str, start: float, end: float, role: str, speech: float, audio: bool = True, broll: bool = False) -> dict:
    return {"name": name, "path": f"/x/{name}", "start": start, "end": end, "dur": end - start, "audio": audio, "broll": broll,
            "facts": {"role": role, "speech": {"ratio": speech}}}


def test_talk_is_never_taken_from_a_broll_files_own_microphone():
    talk, chat = _file("talk.mov", 0, 100, "a_roll", 0.8), _file("C0099.MP4", 100, 130, "voiceover_broll", 0.6, broll=True)
    out, notes = BS.drop_broll_speech([(10.0, 40.0), (105.0, 120.0)], [talk, chat])
    assert out == [(10.0, 40.0)] and notes and "C0099.MP4" in notes[0]
    keep, none = BS.drop_broll_speech([(105.0, 120.0)], [_file("C0099.MP4", 0, 30, "voiceover_broll", 0.6, broll=True)])
    assert keep == [(105.0, 120.0)] and none == []                                    # nothing else to use: the person put them there
    same, _ = BS.drop_broll_speech([(105.0, 120.0)], [talk, chat], keep_sound_ok="*")
    assert same == [(105.0, 120.0)]                                                   # "keep the natural sound" in the notes
    ex, _ = BS.drop_broll_speech([(105.0, 120.0)], [talk, chat], exact=True)
    assert ex == [(105.0, 120.0)]                                                     # an edit list says exactly what plays


def test_natural_sound_is_only_ever_on_when_asked():
    assert BS.allowed_files("a calm video about our week", {}) == set()
    assert BS.allowed_files("please keep the natural sound of the clips", {}) == "*"
    assert BS.allowed_files("", {"broll_sound": ["C0001.MP4"]}) == {"C0001.MP4"}
    bf = _file("C0001.MP4", 0, 20, "broll", 0.01, broll=True)
    c = {"src": 4.0}
    assert BS.jl_mix(c, {}, bf, STYLE, set()) is None and BS.jl_mix(c, {}, bf, STYLE, {"other.mp4"}) is None
    m = BS.jl_mix(c, {}, bf, STYLE, "*")
    assert m and m["gain_db"] == -20.0 and m["ss"] == pytest.approx(4.0 - 0.3)
    talky = _file("C0002.MP4", 0, 20, "broll", 0.5, broll=True)
    assert BS.jl_mix(c, {}, talky, STYLE, "*") is None                                # a shot with talking in it never brings its sound under the voice


def test_a_badly_recorded_stretch_is_flagged_with_its_time():
    class G:
        def grade(self, path, local):
            return {"grade": "bad", "why": "far from the microphone"} if "C0005" in path else {"grade": "good"}
    parts = [{"out": 0.0, "d": 4.0, "a": {"path": "/x/talk.mov", "ss": 0.0}}, {"out": 4.0, "d": 3.0, "a": {"path": "/x/C0005.MP4", "ss": 1.0}},
             {"out": 7.0, "d": 2.0, "a": {"path": "/x/C0005.MP4", "ss": 4.0}}]
    flags = BS.poor_sound_flags(parts, [_file("C0005.MP4", 0, 30, "a_roll", 0.5)], G())
    assert len(flags) == 1 and flags[0]["start"] == 4.0 and flags[0]["end"] == 9.0 and flags[0]["file"] == "C0005.MP4"


def test_apply_jl_marks_a_lead_in_the_piece_before_and_a_tail_in_the_piece_after():
    parts = [{"d": 3.0, "out": 0.0}, {"d": 2.0, "out": 3.0, "a_mix_c": {"path": "/x/n.mp4", "ss": 5.0, "gain_db": -20.0, "lead": 0.3, "tail": 0.3,
                                                                    "first": True, "last": True}}, {"d": 4.0, "out": 5.0}]
    BS.apply_jl(parts)
    assert parts[1]["a_mix"][0]["delay"] == 0.0 and parts[1]["a_mix"][0]["len"] == 2.0
    j, l_ = parts[0]["a_mix"][0], parts[2]["a_mix"][0]
    assert j["ss"] == pytest.approx(4.7) and j["delay"] == pytest.approx(2.7) and j["len"] == pytest.approx(0.3)
    assert l_["ss"] == pytest.approx(7.0) and l_["delay"] == 0.0


# ------------------------------------------------------------------------------------------------ real renders: what the delivered audio contains
def _clip_tone(path: Path, secs: int, color: str, hz: int, vol: float = 0.9) -> None:
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:size=480x270:rate=30:duration={secs}", "-f", "lavfi", "-i",
                    f"sine=frequency={hz}:duration={secs},volume={vol}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(path)], check=True, timeout=120)


def _talk_clip(path: Path, secs: int) -> None:
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=480x270:rate=30:duration={secs}", "-f", "lavfi", "-i",
                    f"sine=frequency=300:duration={secs},volume=0.5", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(path)], check=True, timeout=120)


class _AI(_FakeAI):
    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        if task == "broll_describe":
            return _Res({"scenes": [{"start_s": 0, "end_s": 6, "what": "a plain red brick wall", "quality": 8}]})
        if task == "broll_plan":
            return _Res({"cutaways": [{"at": 4.0, "scene": 0, "length": 2.5, "why": "the wall"}]})
        if task in ("broll_pick", "broll_suitable"):
            return _Res({"scene": -1})
        return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)


def _render(tmp_path, monkeypatch, broll_hz: int, vol: float, notes: str = "") -> Path:
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = tmp_path / "Wall day"
    (folder / "B-roll").mkdir(parents=True)
    _talk_clip(folder / "talk.mov", 14)
    _clip_tone(folder / "B-roll" / "wall.mov", 6, "red", broll_hz, vol)
    said = "We stopped by the old wall today. It was a plain red brick wall and everyone took a photo of it. Then we went home for lunch together.".split()
    words = [{"text": w, "start": 0.4 + i * 0.5, "end": 0.4 + i * 0.5 + 0.35} for i, w in enumerate(said)]
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    monkeypatch.setattr(LP, "LocalAI", lambda c: _AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words if f["name"] == "talk.mov" else [])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Wall day", notes, {"transcriber": "openai", "music": ""}, {"clips": 0})
    projects.prepare_local(proj["id"], {"graphics": "", "music": ""})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done", store.get(proj["job"])
    return max((folder / "Output").glob("*.mp4"), key=lambda p: p.stat().st_mtime)


def _level_at(video: Path, t0: float, t1: float, hz: int, tmp: Path) -> float:
    """Energy (dB) of a pure frequency in a stretch of the delivered audio."""
    wav = tmp / f"a_{hz}_{t0}.wav"
    subprocess.run([FF, "-y", "-v", "error", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True, timeout=60)
    raw = subprocess.run([FF, "-v", "error", "-i", str(wav), "-f", "s16le", "-"], capture_output=True, timeout=60).stdout
    y = np.frombuffer(raw, np.int16).astype(float) / 32768.0
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) / len(y)
    freqs = np.fft.rfftfreq(len(y), 1 / 16000)
    band = spec[(freqs > hz - 25) & (freqs < hz + 25)]
    return float(20 * np.log10(band.max() + 1e-9))


def test_a_broll_shots_own_loud_sound_never_reaches_the_delivered_audio(tmp_path, monkeypatch):
    """The 2026-11-06 defect: a clip filmed as B-roll with its own (camera microphone) sound. Here 1 kHz at 0.9 under a 300 Hz voice."""
    out = _render(tmp_path, monkeypatch, 1000, 0.9)
    voice = _level_at(out, 4.3, 6.2, 300, tmp_path)                                    # during the cutaway (4.0-6.5 s) the speaker's sound keeps playing
    leak = _level_at(out, 4.3, 6.2, 1000, tmp_path)
    assert voice > -40 and leak < voice - 35, (voice, leak)                            # the shot's tone is not in there at all


def test_nat_sound_when_asked_is_a_j_cut_about_twenty_db_under_the_voice(tmp_path, monkeypatch):
    out = _render(tmp_path, monkeypatch, 60, 0.9, notes="Keep the natural sound of the clips.")      # (60 Hz hum: under the speech band, so it counts as ambience)
    voice = _level_at(out, 4.3, 6.2, 300, tmp_path)
    amb = _level_at(out, 4.3, 6.2, 60, tmp_path)
    assert voice > -40 and voice - 32 < amb < voice - 8, (voice, amb)                  # present, and well under the voice (about -20 dB)
