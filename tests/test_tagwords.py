"""Keywords for footage that is only searchable so far (tagwords.py): the vocabulary and its cached matrix, the top three words of a file
from its embeddings (near-synonyms folded, weak matches dropped, the library-wide average taken off), keywords kept on each record at
index time and made later for older records, no model = the folder and file name, and a description sharpening them.
The picture model is faked: every word (and every picture) is a seeded unit vector in 256 dimensions, so cosines are exact and repeatable."""

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipfarm.clipper import catalog as C
from clipfarm.clipper import tagwords as TW
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

DIM = 256


def vec(word: str) -> np.ndarray:
    seed = int(hashlib.sha1(word.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


def unit(v):
    return (v / np.linalg.norm(v)).astype("float32")


class Concepts:
    """Words -> seeded vectors ('a photo of gym' and 'gym' are the same idea). A picture is whatever vector it is given (tests build them);
    the colour of a real test video stands for a word: red = gym, green = parade, blue = night."""
    name = "concepts"

    def __init__(self):
        self.text_calls = 0

    def text(self, texts):
        self.text_calls += 1
        return np.array([vec(re_word(t)) for t in texts], dtype="float32")

    def images(self, ims):
        out = []
        for im in ims:
            a = np.asarray(im.convert("RGB"), dtype="float32").reshape(-1, 3).mean(0)
            word = ("gym", "parade", "night")[int(np.argmax(a))]
            out.append(vec(word))
        return np.array(out, dtype="float32")

    def chance(self, score):
        return max(0.0, float(score) - 0.2) / 0.8 * 0.2     # (cosine 1.0 -> 0.2: a clear match; the noise of random vectors (< 0.2) -> 0, like SigLIP's sigmoid)


def re_word(prompt: str) -> str:
    w = prompt.replace("a photo of ", "").strip()
    return w


def make_tagger(tmp_path, emb=None, model="concepts"):
    return TW.TagWords(tmp_path / "cat", emb or Concepts(), model)


# ------------------------------------------------------------------------------------------------------------- vocabulary
def test_the_vocabulary_is_about_four_hundred_distinct_plain_words():
    words = TW.vocabulary()
    assert 330 <= len(words) <= 520 and len(set(words)) == len(words)
    shown = [TW.shown(w) for w in words]
    for must in ("students", "classroom", "walking", "youth camp", "parade", "night", "graduation", "close-up", "campus", "stage"):
        assert must in shown, must
    assert not any(s.startswith(("a ", "an ")) for s in shown)


def test_the_matrix_is_made_once_kept_and_rebuilt_when_the_model_or_the_words_change(tmp_path, monkeypatch):
    emb = Concepts()
    t1 = make_tagger(tmp_path, emb)
    M = t1.matrix()
    assert M.shape == (len(TW.vocabulary()), DIM) and emb.text_calls >= 1
    assert (tmp_path / "cat" / "tagwords.npy").exists() and (tmp_path / "cat" / "tagwords.json").exists()
    emb.text_calls = 0
    t2 = make_tagger(tmp_path, emb)                                   # same model, same words: straight from disk
    assert t2.matrix().shape == M.shape and emb.text_calls == 0
    t3 = make_tagger(tmp_path, emb, model="another-model")            # another picture model: embedded again
    t3.matrix()
    assert emb.text_calls >= 1
    emb.text_calls = 0
    monkeypatch.setattr(TW, "vocabulary", lambda: ["one", "two", "three"])
    t4 = make_tagger(tmp_path, emb, model="another-model")            # another vocabulary: embedded again
    t4.matrix()
    assert emb.text_calls >= 1


def test_no_embedder_and_no_cache_means_no_matrix_and_no_words(tmp_path):
    t = TW.TagWords(tmp_path / "cat", None, "concepts")
    assert t.matrix() is None and t.pick(np.zeros((2, DIM), "float32")) == []


# ------------------------------------------------------------------------------------------------------------------ picking
def frames_of(*words, noise=0.05, n=4, seed=1):
    base = unit(sum(vec(TW.shown(w)) for w in words))             # (the matrix embeds the shown word: no 'a ' in front)
    rng = np.random.default_rng(seed)
    return np.array([unit(base + noise * rng.normal(size=DIM).astype("float32") / np.sqrt(DIM)) for _ in range(n)], dtype="float32")


def test_the_top_words_are_the_ones_the_pictures_show(tmp_path):
    t = make_tagger(tmp_path)
    tags = t.pick(frames_of("a classroom", "students", "listening"))
    assert len(tags) == 3 and set(tags) == {"classroom", "students", "listening"}


def test_near_synonyms_count_as_one_idea(tmp_path):
    t = make_tagger(tmp_path)
    # 'a student' and 'students' are the same words in different forms: only one of them is offered
    tags = t.pick(frames_of("students", "a student", "a library"))
    assert not ({"students", "student"} <= set(tags))
    assert "library" in tags
    # one word inside another ('walking' / 'a group walking'-like) is one idea too
    assert TW._pairs_ok("people walking", ["walking"]) is False and TW._pairs_ok("night", ["walking"]) is True


def test_a_picture_that_shows_nothing_in_the_list_gets_no_words(tmp_path):
    t = make_tagger(tmp_path)
    noise = unit(np.random.default_rng(99).normal(size=DIM).astype("float32"))
    assert t.pick(np.array([noise] * 3)) == []                       # no word is near it: better no keyword than a wrong one


def test_fewer_than_three_words_when_only_one_is_a_clear_match(tmp_path):
    t = make_tagger(tmp_path)
    tags = t.pick(frames_of("a parade", noise=0.0))
    assert tags[0] == "parade" and len(tags) <= 3


def test_a_word_that_fits_everything_gives_way_to_one_that_fits_this_file(tmp_path, monkeypatch):
    monkeypatch.setattr(TW, "IDF_AFTER", 40)
    t = make_tagger(tmp_path)
    for k in range(60):                                              # a library where nearly every file shows 'outdoors' (and one other thing)
        t.pick(frames_of("outdoors", ("a gym", "a library", "a stage", "a street")[k % 4], seed=k), count=True)
    both = frames_of("outdoors", "a gym", noise=0.0)
    raw = TW.TagWords(tmp_path / "other", Concepts(), "concepts")
    assert raw.pick(both, count=False)[0] in ("outdoors", "gym")     # (before the average is known the two are close)
    assert t.pick(both, count=False)[0] == "gym"                     # with the average taken off, the specific one leads
    assert "outdoors" not in t.pick(both, count=False)[:1]


def test_statistics_are_kept_between_runs(tmp_path):
    t = make_tagger(tmp_path)
    for k in range(30):
        t.pick(frames_of("a gym", seed=k))
    t.flush()
    t2 = make_tagger(tmp_path)
    t2.matrix()
    assert t2._load_stats()["n"] == 30


# ------------------------------------------------------------------------------------------------ names when there is no model
def test_with_no_model_the_folder_and_file_name_give_the_words():
    assert TW.path_words("Youth Camp/Day 2/IMG_0042 canoe race.mov") == ["canoe", "race", "youth"]
    assert TW.path_words("DCIM/100MSDCF/C0018.MP4") == []
    assert TW.path_words("Photo/Archive/Lightroom Originals/IMG_8377.CR3")[:1] == ["lightroom"]
    assert TW.path_words("") == [] and TW.path_words("////") == []                           # (never raises on odd names)


# ---------------------------------------------------------------------------------------------------------- the description
def test_a_description_sharpens_the_keywords_by_rarity(tmp_path):
    t = make_tagger(tmp_path)
    for k in range(60):
        t.note_description("people standing in a hallway" if k % 2 else "a person walking outside a building")
    merged = t.merge(["students", "walking", "hallway"], ["A woman holding a trophy at a graduation ceremony"])
    assert len(merged) == 3 and ("trophy" in merged or "graduation" in merged or "ceremony" in merged)
    assert "students" in merged                                      # the strongest picture word stays
    # a word the pictures and the description agree on goes up ('students' / 'student')
    agree = t.merge(["walking", "students"], ["a student walking"])
    assert agree[0] in ("walking", "students") and len(agree) <= 3


def test_early_on_the_description_does_not_push_picture_words_out(tmp_path):
    t = make_tagger(tmp_path)
    assert t.merge(["gym", "stage", "night"], ["a man dribbles a basketball"]) == ["gym", "stage", "night"]


# --------------------------------------------------------------------------------------------------------- in the catalogue
def _video(path: Path, color: str, secs: int = 7):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={secs}:r=25",
                    "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", str(path)], check=True)


def _cat(tmp_path, root, emb="default"):
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["catalog"] = {"roots": [str(root)], "dir": str(tmp_path / "cat"), "frame_every_s": 2.0, "max_frames": 50, "min_score": 0.5, "workers": 1}
    return C.Catalog(cfg, embedder=Concepts() if emb == "default" else emb)


def test_a_file_gets_its_keywords_when_it_is_catalogued_and_the_live_event_carries_them(tmp_path):
    root = tmp_path / "Camp" / "Day 1"
    _video(root / "clip.mp4", "red")
    cat = _cat(tmp_path, tmp_path / "Camp")
    cat.index()
    rec = next(cat.records())
    assert rec["tags"] and rec["tags"][0] == "gym" and len(rec["tags"]) <= 3
    done = [e for e in cat.live_since(0)["events"] if e["kind"] == "done" and e.get("step") == "frames"]
    assert done and done[-1]["tags"] == rec["tags"]


def test_with_no_picture_model_loaded_the_names_stand_in(tmp_path):
    root = tmp_path / "Camp" / "Canoe Race"
    _video(root / "clip.mp4", "red")
    cat = _cat(tmp_path, tmp_path / "Camp")
    cat._embedder = Concepts()
    cat.index()
    rec = next(cat.records())
    cat2 = _cat(tmp_path, tmp_path / "Camp", emb=None)                # a catalogue whose model is not loaded
    assert cat2.tagger() is None
    assert cat2.picture_tags(np.zeros((1, DIM), "float32"), "Camp/Canoe Race/clip.mp4") == ["canoe", "race", "camp"]
    assert rec["tags"]


def test_older_records_get_keywords_from_their_embeddings_without_reading_video(tmp_path, monkeypatch):
    root = tmp_path / "Camp"
    _video(root / "a.mp4", "green")
    cat = _cat(tmp_path, root)
    cat.index()
    rec = next(cat.records())
    old = {k: v for k, v in rec.items() if k != "tags"}               # made before keywords existed
    cat.save(old)
    monkeypatch.setattr(ffmpeg, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a video was read")))
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a process was started")))
    assert cat.fill_tags() == 1
    assert next(cat.records())["tags"][0] == "parade"
    assert cat.fill_tags() == 0                                       # nothing left to do


def test_fill_tags_never_loads_the_model(tmp_path):
    root = tmp_path / "Camp"
    _video(root / "a.mp4", "blue")
    cat = _cat(tmp_path, root)
    cat.index()
    rec = next(cat.records())
    cat.save({k: v for k, v in rec.items() if k != "tags"})
    fresh = _cat(tmp_path, root)
    fresh._embedder = None
    fresh.embedder = lambda: (_ for _ in ()).throw(AssertionError("the model was loaded"))
    assert fresh.fill_tags() == 0


def test_search_results_carry_keywords_even_for_files_catalogued_before(tmp_path):
    root = tmp_path / "Camp"
    _video(root / "a.mp4", "red")
    _video(root / "b.mp4", "blue")
    cat = _cat(tmp_path, root)
    cat.index()
    for rec in list(cat.records()):                                   # as if made before keywords
        cat.save({k: v for k, v in rec.items() if k != "tags"})
    cat.compile()
    shots = cat.search("gym", n=5)
    assert shots and all(isinstance(s["tags"], list) for s in shots)
    top = next(s for s in shots if s["rel"].endswith("a.mp4"))
    assert top["tags"] and top["tags"][0] == "gym"
    assert next(r for r in cat.records() if r["rel"].endswith("a.mp4")).get("tags")        # and kept on the record


def test_a_description_keeps_the_picture_words_and_adds_rare_ones(tmp_path):
    root = tmp_path / "Camp"
    _video(root / "a.mp4", "red")
    cat = _cat(tmp_path, root)
    cat.index()
    rec = next(cat.records())
    for k in range(30):
        cat._tagger.note_description("people standing in a hallway")
    tags = cat._described_tags(rec, ["a boy dunking a basketball above the rim"])
    assert tags[0] == "gym" and len(tags) <= 3 and any(w in tags for w in ("basketball", "dunking", "above", "rim"))
    # the sharpened words are what a later re-index of the unchanged file keeps
    cat.save({**rec, "tags": tags, "described": "x", "scenes": [{"start": 0, "end": 5, "what": "x", "quality": 5}]})
    assert next(cat.records())["tags"] == tags
