"""Stock search that matches (2026-11-06: "you search for something and it is not at all related"): extra AI searches, pictures ranked
by SigLIP, songs ranked by listening (CLAP), no singing unless asked, tiny/short/duplicate results out. Providers and models are fakes."""
import json
import time

import numpy as np
import pytest

from clipfarm.clipper import musicsources as MS
from clipfarm.clipper import stockfind as SF


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for d in (SF._CACHE, SF._VEC, SF._ALT, SF._PENDING, SF._ITEMS, MS._AVEC, MS._cache):
        d.clear()
    monkeypatch.setattr(SF, "_REWRITE", None)
    monkeypatch.setattr(SF, "_EMBED", None)
    monkeypatch.setattr(MS, "_CLAP", None)


def item(i, what, src="openverse", kind="photo", w=1920, h=1080, dur=0.0, creator="x"):
    return {"source": src, "kind": kind, "id": f"{src}:{kind}:{i}", "what": what, "thumb": f"https://t/{i}.jpg", "preview": "", "w": w, "h": h,
            "dur": dur, "page": "", "license": "CC0", "credit": "", "download": f"https://d/{i}.jpg", "creator": creator}


class Pics:
    """A stand-in SigLIP: a picture's vector says how 'talking' it is (by its thumbnail name)."""
    TALK = {"1": 0.18, "2": 0.15, "3": 0.02, "4": 0.12, "5": 0.01, "6": 0.11, "7": 0.10, "8": 0.09, "9": 0.181}

    def images(self, ims):                                                              # (each picture its own direction, as in
        out = []                                                                        #  SigLIP's 768 dimensions; 9 is a copy of 1)
        for im in ims:
            s = self.TALK.get(im, 0.0)
            v = np.zeros(12)
            v[0], v[1 + int({"9": "1"}.get(im, im)) % 11] = s, (1 - s * s) ** 0.5
            out.append(v)
        return np.array(out)

    def text(self, texts):
        return np.array([[1.0] + [0.0] * 11])


def test_pictures_are_ordered_by_what_they_show_and_the_unrelated_go(monkeypatch):
    monkeypatch.setattr(SF, "_thumb_image", lambda url: url.rsplit("/", 1)[-1].split(".")[0])
    SF.configure(embedder=lambda: Pics())
    items = [item(i, w) for i, w in [(3, "Portrait of the Artist"), (1, "two people talking"), (5, "a meme"), (2, "bicycle conversation"),
                                      (4, "George and Ting"), (6, "a"), (7, "b"), (8, "c")]]
    out = SF.rank("two people talking", items)
    assert [x["what"] for x in out[:2]] == ["two people talking", "bicycle conversation"]
    assert "Portrait of the Artist" not in [x["what"] for x in out] and "a meme" not in [x["what"] for x in out]   # below the floor
    assert out[0]["best"] and out[0]["match"] == 0.18 and not out[-1]["best"]


def test_a_few_results_always_stay_and_the_same_picture_shows_once(monkeypatch):
    monkeypatch.setattr(SF, "_thumb_image", lambda url: url.rsplit("/", 1)[-1].split(".")[0])
    SF.configure(embedder=lambda: Pics())
    items = [item(i, f"thing {i}") for i in (3, 5, 1, 9)]                                # 9 looks just like 1 (a re-upload)
    out = SF.rank("two people talking", items, keep_least=3)
    assert len(out) == 3 and [x["what"] for x in out].count("thing 9") + [x["what"] for x in out].count("thing 1") == 1


def test_without_the_picture_model_it_is_the_providers_order_small_last():
    items = [item(1, "small", w=800, h=600), item(2, "big"), item(3, "big too")]
    out = SF.rank("x", items)
    assert [x["what"] for x in out] == ["big", "big too", "small"] and not any(x["best"] for x in out)


def test_search_merges_the_ai_searches_skips_openverse_for_them_and_drops_bad_results(monkeypatch):
    calls = []

    def pex(q, n, page):
        calls.append(("pexels", q))
        return ([item(f"p{q[:3]}{k}", f"{q} {k}", src="pexels", kind="video", dur=8.0) for k in range(2)]
                + [item(f"short{q[:3]}", "too short", src="pexels", kind="video", dur=2.0)])

    def ov(q, n, page):
        calls.append(("openverse", q))
        return [item("o1", "Student Studying (11056363313)"), item("o2", "Student Studying (11056201595)"), item("o3", "tiny", w=375, h=500)]
    ov.slow = True                                                                      # (as Openverse is marked)
    monkeypatch.setattr(SF, "PROVIDERS", [("video", pex), ("photo", ov)])
    SF.configure(rewrite=lambda q, k: ['["friends chatting", "candid conversation"]'])        # (the JSON-in-a-string answer)
    out = SF.search("two people talking", ("video", "photo"), n=30)
    assert ("openverse", "two people talking") in calls and not any(s == "openverse" and q != "two people talking" for s, q in calls)
    assert {q for s, q in calls if s == "pexels"} == {"two people talking", "friends chatting", "candid conversation"}
    whats = [x["what"] for x in out]
    assert "too short" not in whats and "tiny" not in whats
    assert sum("Student Studying" in w for w in whats) == 1                            # (the same photo, uploaded twice)


def test_a_slow_ai_never_holds_the_search_up(monkeypatch):
    SF.configure(rewrite=lambda q, k: (time.sleep(0.5), ["later"])[1])
    t = time.time()
    assert SF.searches_for("dogs", wait=0.05) == ["dogs"] and time.time() - t < 0.4
    time.sleep(0.7)
    assert SF.searches_for("dogs", wait=0.0) == ["dogs", "later"]                       # kept for "More" and next time
    SF.configure(rewrite=lambda q, k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert SF.searches_for("cats") == ["cats"]


def test_music_leaves_out_singing_unless_asked_and_listens_to_the_best(monkeypatch):
    def song(i, title, tags):
        return {"source": "openverse", "id": str(i), "title": title, "artist": "a", "dur": 120, "preview": f"https://m/{i}.mp3",
                "license": "CC0", "license_url": "", "credit": "", "page": "", "tags": tags, "provider": "jamendo", "filetype": "mp3"}
    raw = [song(1, "Harsh Calm [Hardcore Techno]", ["gabber", "instrumental"]), song(2, "Jabberhead", ["female", "vocal", "calm"]),
           song(3, "Deep Study of a Calm Mind", ["piano", "calm"]), song(4, "Calm Space", ["calm"]), song(5, "Soft keys", ["piano"]),
           song(6, "Night", ["calm"]), song(7, "Rain", ["piano"])]
    monkeypatch.setattr(MS, "_openverse_get", lambda params: [{}] * 0)
    monkeypatch.setattr(MS, "_openverse_track", lambda r: r)
    monkeypatch.setattr(MS, "_openverse_get", lambda params: list(raw) if params.get("q") else [])

    class Clap:
        SOUND = {"1": [0.0, 1.0], "3": [1.0, 0.0], "5": [0.9, 0.1], "7": [0.8, 0.2]}

        def audio(self, path, start, dur):
            v = np.array(self.SOUND.get(str(path), [0.3, 0.3]), float)
            return v / np.linalg.norm(v)

        def text(self, texts):
            return np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])

    class FakeExcerpt:
        def __init__(self, url):
            self.id = url.rsplit("/", 1)[-1].split(".")[0]

        def __enter__(self):
            return self.id

        def __exit__(self, *a):
            pass
    monkeypatch.setattr(MS, "excerpt", FakeExcerpt)
    MS.configure(clap=lambda: Clap())
    out = MS.openverse_search("calm piano", n=10)
    titles = [t["title"] for t in out]
    assert "Jabberhead" not in titles                                                  # singing: not for background music
    assert titles[0] == "Deep Study of a Calm Mind" and out[0]["best"]
    assert titles.index("Harsh Calm [Hardcore Techno]") > 2
    MS._cache.clear()
    assert "Jabberhead" in [t["title"] for t in MS.openverse_search("calm female vocals", n=10)]


def test_music_without_the_listening_model_keeps_the_tag_order(monkeypatch):
    tracks = [{"id": "1", "title": "a"}, {"id": "2", "title": "b"}]
    assert MS.listen_rank("x", tracks) == tracks


def test_a_song_slice_is_fetched_by_range_never_the_whole_song(monkeypatch):
    seen = []

    class R:
        def __init__(self, status, headers, content=b""):
            self.status_code, self.headers, self.content = status, headers, content

        def raise_for_status(self):
            pass
    monkeypatch.setattr(MS.requests, "get", lambda url, headers=None, timeout=None: seen.append(headers["Range"]) or (
        R(206, {"Content-Range": "bytes 0-1/5000000"}) if headers["Range"] == "bytes=0-1" else R(206, {}, b"x" * 600_000)))
    with MS.excerpt("https://m/1.mp3") as f:
        assert f.exists() and f.stat().st_size == 600_000
    assert seen == ["bytes=0-1", "bytes=2000000-2599999"] and not f.exists()
    with pytest.raises(MS.MusicSourceError):
        MS.excerpt("http://plain/1.mp3").__enter__()


def test_the_music_model_is_one_whose_words_work():
    """laion/larger_clap_music gave every phrase the same vector; the library and stock ranking use clap-htsat-unfused."""
    from clipfarm.clipper import musiclib, studio
    assert musiclib.MODEL == "laion/clap-htsat-unfused" == studio.MUSIC_MODEL and musiclib.VERSION >= 3
