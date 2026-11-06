"""Names spelled the person's way (Settings › How it edits › Names and words), and no cut that flips a sentence's meaning."""
import json
from pathlib import Path

from clipfarm.clipper import glossary as G
from clipfarm.clipper import negation as N

TT = Path("/private/tmp/claude-503/-Volumes-T7-Coding-Clipfarm/c4200676-e79f-4af6-9542-2fc5bab9ca6d/scratchpad/tt_transcript.json")


def W(text, s, e=None):
    return {"text": text, "start": float(s), "end": float(e if e is not None else s + 0.3)}


def words_of(line, t0=0.0, step=0.4):
    return [W(x, t0 + i * step, t0 + i * step + 0.3) for i, x in enumerate(line.split())]


# ---------------------------------------------------------------------------------------------- glossary
def test_the_real_misspellings_of_northfield_are_fixed():
    heard = [W("At", 0), W("Northfeld,", 1), W("faith", 2), W("Nortfield,", 3), W("Et", 4), W("Northfild,", 5), W("northfield", 6)]
    out = G.correct(heard, ["Northfield"])
    assert [w["text"] for w in out] == ["At", "Northfield,", "faith", "Northfield,", "Et", "Northfield,", "Northfield"]
    assert [(w["start"], w["end"]) for w in out] == [(w["start"], w["end"]) for w in heard]          # timing kept


def test_ordinary_words_and_other_names_are_left_alone():
    for line in ("a must", "Emma said so", "famous emails", "Amass the team", "I am us"):
        ws = words_of(line)
        assert G.correct(ws, ["Northfield"]) == ws, line


def test_multi_word_names_keep_each_words_timing_and_a_split_name_is_merged():
    out = G.correct(words_of("we went to tamsin thorne delacroix, again"), ["Tamsin Thorne Delacroix"])
    assert [w["text"] for w in out] == ["we", "went", "to", "Tamsin", "Thorne", "Delacroix,", "again"]
    out = G.correct(words_of("Tamsen Thorne Delacroix"), ["Tamsin Thorne Delacroix"])
    assert [w["text"] for w in out] == ["Tamsin", "Thorne", "Delacroix"]
    merged = G.correct([W("Mc", 0, 0.2), W("Kinsey", 0.25, 0.6), W("said", 0.7)], ["McKinsey"])
    assert merged[0] == {"text": "McKinsey", "start": 0.0, "end": 0.6} and merged[1]["text"] == "said"


def test_the_names_list_is_cleaned_and_reaches_the_prompts():
    assert G.clean(["  Northfield ", "northfield", "", "Tamsin   Thorne Delacroix\nLakeside Arts Academy", "123"]) == ["Northfield", "Tamsin Thorne Delacroix", "Lakeside Arts Academy"]
    assert "Northfield" in G.prompt_note(["Northfield"]) and G.prompt_note([]) == ""


def test_on_the_work_macs_take_every_northfield_is_spelled_right():
    if not TT.exists():
        return
    words = json.loads(TT.read_text())["words"]
    out = G.correct(words, ["Northfield"])
    assert len(out) == len(words)
    assert not any(t in " ".join(w["text"] for w in out) for t in ("Northfeld", "Nortfield", "Northfild"))


# ---------------------------------------------------------------------------------------------- negations
def test_a_cut_never_drops_not_from_a_kept_sentence():
    ws = words_of("I did not like it.")                                  # I 0-.3, did .4-.7, not .8-1.1, like 1.2-1.5, it. 1.6-1.9
    ivs, notes = N.keep_negations([(0.0, 0.75), (1.15, 1.95)], ws)      # "not" cut out
    assert N.lost_negations(ivs, ws) == [] and notes and "not" in notes[0]
    assert any(a <= 0.95 <= b for a, b in ivs)                          # "not" plays


def test_a_filler_next_to_not_still_goes():
    ws = [W("I", 0.0), W("did", 0.4), W("um", 0.8), W("not", 1.2), W("like", 1.6), W("it.", 2.0)]
    ivs = [(0.0, 0.75), (1.15, 2.35)]                                     # the "um" cut, "not" kept
    out, notes = N.keep_negations(ivs, ws)
    assert out == ivs and notes == []


def test_a_whole_sentence_left_out_is_fine_and_a_far_negation_drops_the_sentence():
    ws = words_of("This one goes. I did not like it.")
    ivs = [(0.0, 1.1)]                                                    # only the first sentence
    assert N.keep_negations(ivs, ws) == (ivs, [])
    far = [W("We", 0), W("never", 0.4), W("um", 1.0), W("said", 3.5), W("goodbye", 3.9), W("at", 4.3), W("all.", 4.7)]
    other = W("Interviewer:", 2.0)
    out, notes = N.keep_negations([(3.4, 5.1)], far + [other])           # "We never" far away behind someone else's word: whole sentence out
    assert out == [] or all(b <= 3.4 or a >= 5.1 for a, b in out)
    assert notes and "left out" in notes[0]


def test_contractions_count_and_order_is_kept():
    ws = words_of("We won't stop now.", t0=10)
    ivs = [(20.0, 22.0), (10.0, 10.35), (10.75, 11.6)]                    # reordered story: "won't" cut
    out, _ = N.keep_negations(ivs, ws)
    assert out[0] == (20.0, 22.0) and N.lost_negations(out, ws) == []


# ---------------------------------------------------------------------------------------------- wiring
def test_the_setting_is_cleaned_and_kept(tmp_path, monkeypatch):
    from clipfarm.clipper import studio as SD
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path))
    s = SD.Settings(tmp_path)
    assert s.get()["glossary"] == []
    out = s.update({"glossary": ["Northfield", " northfield ", "", "Tamsin Thorne Delacroix"]})
    assert out["glossary"] == ["Northfield", "Tamsin Thorne Delacroix"]
    G._cache.update(at=0.0, val=None)
    assert G.terms() == ["Northfield", "Tamsin Thorne Delacroix"]                              # read fresh from the app's settings
    G._cache.update(at=0.0, val=None)


def test_parakeet_words_come_back_with_the_names_fixed(tmp_path, monkeypatch):
    import subprocess
    from clipfarm.clipper import parakeet as P
    exe = tmp_path / "parakeet-mlx"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(P, "binary", lambda cfg=None: exe)
    monkeypatch.setattr(P.ffmpeg, "run", lambda *a, **k: None)
    monkeypatch.setattr(P.ffmpeg, "duration", lambda p: 10.0)

    def run(cmd, **kw):
        out = Path(cmd[cmd.index("--output-dir") + 1]) / "audio.json"
        out.write_text(json.dumps({"sentences": [{"tokens": [{"text": " At", "start": 0.1, "end": 0.3}, {"text": " Northfeld,", "start": 0.3, "end": 0.8}]}]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(P.subprocess, "run", run)
    monkeypatch.setattr(G, "terms", lambda cfg=None: ["Northfield"])
    assert [w["text"] for w in P.transcribe(tmp_path / "a.wav")] == ["At", "Northfield,"]


def test_the_ai_is_told_the_spellings_when_it_writes(monkeypatch, tmp_path):
    from clipfarm.clipper import localai as LA
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "fallback": "", "data_dir": "ai"})
    cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "fallback": False})
    ai = LA.LocalAI(cfg)
    seen = []
    monkeypatch.setattr(ai.claude.__class__, "available", lambda self: True)
    monkeypatch.setattr(ai.claude.__class__, "complete_json", lambda self, system, *a, **k: (seen.append(system) or ({"ok": 1}, "sonnet")))
    monkeypatch.setattr(G, "terms", lambda cfg=None: ["Northfield"])
    ai.complete_json("video_title", "Write a title.", "p", {})
    assert "Northfield" in seen[-1] and "Spell these names exactly" in seen[-1]
