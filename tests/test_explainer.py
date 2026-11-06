"""Tester's checks for the explainer pipeline's deterministic parts (card_spec, flow cap, picture menu, downloads, look rotation, card fallbacks) and
the new flow / bars / line cards and page styles. Fakes only: no network, no AI, no ElevenLabs.

Tests marked xfail are DEFECTS found by the tester (the reason says what is wrong); they are not strict so a fix turns them into XPASS."""
import io
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import explainer as EX
from clipfarm.clipper import gfxcards as G
from clipfarm.clipper.graphics import THEMES
from clipfarm.core import ffmpeg, tts

WORDS = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate("first we mix then we heat and finally we cool it down slowly".split())]


# ------------------------------------------------------------------------------------------------ card_spec
def beat(**v):
    return {"narration": "x " * 10, "visual": v}


def test_card_spec_stat_needs_digits_and_label():
    assert EX.card_spec(beat(kind="stat", value="1939", label="the year"), 6, [])["kind"] == "bigstat"
    assert EX.card_spec(beat(kind="stat", value="six", label="the year"), 6, []) is None
    assert EX.card_spec(beat(kind="stat", value="1939", label=""), 6, []) is None
    assert EX.card_spec(beat(kind="stat", label="x"), 6, []) is None


def test_card_spec_stat_value_limit_matches_prompt():
    assert EX.card_spec(beat(kind="stat", value="1,234,567", label="people"), 6, []) is None


def test_card_spec_flow_needs_three_steps_and_a_title():
    assert EX.card_spec(beat(kind="flow", title="T", steps=["a", "b"]), 6, []) is None
    assert EX.card_spec(beat(kind="flow", title="T", steps=["a", "", "b", ""]), 6, []) is None          # blanks do not count
    assert EX.card_spec(beat(kind="flow", title="", steps=["a", "b", "c"]), 6, []) is None
    s = EX.card_spec(beat(kind="flow", title="T", steps=list("abcdefg")), 6, [])
    assert s["kind"] == "flow" and len(s["steps"]) == 5 and "times" not in s                              # capped at five, no words -> no times


def test_card_spec_flow_times_follow_the_spoken_words():
    s = EX.card_spec(beat(kind="flow", title="T", steps=["mix", "heat", "cool"]), 8, WORDS)
    assert s["times"] == sorted(s["times"]) and len(s["times"]) == 3


def test_card_spec_bars_and_timeline_and_slide():
    assert EX.card_spec(beat(kind="bars", title="T", labels=["a"], values=["1"]), 6, []) is None
    assert EX.card_spec(beat(kind="bars", title="T", labels=["a", "b"], values=["1"]), 6, []) is None
    assert EX.card_spec(beat(kind="bars", title="T", labels=["a", "b"], values=["1", "2"]), 6, [])["kind"] == "bars"
    assert EX.card_spec(beat(kind="timeline", title="T", events=[{"date": "1", "label": "a"}] * 2), 6, []) is None
    assert EX.card_spec(beat(kind="timeline", title="T", events=[{"date": "1", "label": "a"}] * 6), 6, [])["events"].__len__() == 5
    assert EX.card_spec(beat(kind="timeline", title="T", events=[{"date": "1", "label": "a"}] * 3), 6, [])["kind"] == "line"
    assert EX.card_spec(beat(kind="slide", title="T", bullets=["only one"]), 6, []) is None
    assert EX.card_spec(beat(kind="slide", title="T", bullets=list("abcdef")), 6, [])["bullets"] == list("abcd")
    assert EX.card_spec(beat(kind="photo", image="i1"), 6, []) is None


def test_card_spec_visual_without_kind_does_not_crash():
    assert EX.card_spec(beat(title="x"), 6, []) is None


# ------------------------------------------------------------------------------------------------ write_script
class FakeAI:
    def __init__(self, data):
        self.data = data

    def complete_json(self, *a, **k):
        return SimpleNamespace(data=json.loads(json.dumps(self.data)))


def flow_beat(i):
    return {"narration": f"step {i} happens and then another thing follows it", "visual": {"kind": "flow", "title": f"Flow {i}", "steps": ["a", "b", "c", "d", "e"]}}


def test_write_script_turns_flows_beyond_four_into_slides_and_drops_short_beats():
    data = {"title": "T", "beats": [flow_beat(i) for i in range(7)] + [{"narration": "too short", "visual": {"kind": "photo"}}]}
    sc = EX.write_script(FakeAI(data), "topic", [{"title": "A", "text": "t"}], 6, [{"id": "i1", "desc": "d"}])
    kinds = [b["visual"]["kind"] for b in sc["beats"]]
    assert kinds == ["flow"] * 4 + ["slide"] * 3
    assert sc["beats"][4]["visual"]["bullets"] == ["a", "b", "c", "d"] and sc["beats"][4]["visual"]["title"] == "Flow 4"


def test_write_script_empty_ai_answer_gives_no_beats_not_a_crash():
    assert EX.write_script(FakeAI({"title": "T"}), "t", [{"title": "A", "text": "t"}], 6, [])["beats"] == []


def test_write_script_beat_without_visual_does_not_crash():
    data = {"title": "T", "beats": [{"narration": "one two three four five six seven"}]}
    EX.write_script(FakeAI(data), "t", [{"title": "A", "text": "t"}], 6, [])


def test_slide_made_from_a_flow_with_too_few_steps_falls_through_to_a_photo():
    data = {"title": "T", "beats": [flow_beat(i) for i in range(5)]}
    data["beats"][4]["visual"]["steps"] = ["only one"]
    sc = EX.write_script(FakeAI(data), "t", [{"title": "A", "text": "t"}], 6, [])
    assert EX.card_spec(sc["beats"][4], 5, []) is None


# ------------------------------------------------------------------------------------------------ pictures menu
def page(title, lic="Public domain", mime="image/jpeg", w=1600, h=900, thumb=True, desc="A gearbox"):
    info = {"mime": mime, "width": w, "height": h, "descriptionurl": "https://c/" + title, "extmetadata": {"LicenseShortName": {"value": lic},
            "ImageDescription": {"value": desc}, "Artist": {"value": "<a>Jane</a>"}}}
    if thumb:
        info["thumburl"] = "https://t/" + title
    return {"title": title, "imageinfo": [info]}


def test_info_batch_licence_size_and_name_filters(monkeypatch):
    pages = [page("File:ok.jpg"), page("File:ccby.jpg", "CC BY 4.0"), page("File:bysa.jpg", "CC BY-SA 3.0"), page("File:cc0.png", "CC0", "image/png"),
             page("File:nc.jpg", "CC BY-NC 4.0"), page("File:nd.jpg", "CC BY-ND 2.0"), page("File:fair.jpg", "Fair use"), page("File:none.jpg", ""),
             page("File:small.jpg", w=500, h=300), page("File:thin.jpg", w=2000, h=400), page("File:logo of x.jpg"), page("File:Flag of Y.jpg"),
             page("File:map.jpg"), page("File:anim.gif", mime="image/gif"), page("File:nothumb.jpg", thumb=False), page("File:d.svg", mime="image/svg+xml", w=0, h=0),
             {"title": "File:missing.jpg", "missing": True}]
    monkeypatch.setattr(EX.WK, "api_commons", lambda p: {"query": {"pages": pages}})
    got = {i["title"]: i for i in EX._info_batch(["x"])}
    assert set(got) == {"File:ok.jpg", "File:ccby.jpg", "File:bysa.jpg", "File:cc0.png", "File:d.svg"}
    assert got["File:ok.jpg"]["artist"] == "Jane" and got["File:ok.jpg"]["desc"] == "A gearbox"


def test_info_batch_description_falls_back_to_file_name_and_strips_html(monkeypatch):
    p = page("File:Planetary gear set.jpg", desc="")
    q = page("File:b.jpg", desc="<b>Bold</b>   text\n here")
    monkeypatch.setattr(EX.WK, "api_commons", lambda params: {"query": {"pages": [p, q]}})
    a, b = EX._info_batch(["x"])
    assert a["desc"] == "Planetary gear set" and b["desc"] == "Bold text here"


def test_info_batch_chunks_of_forty_and_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(EX.WK, "api_commons", lambda p: (calls.append(p["titles"].count("|") + 1), {"query": {}})[1])
    assert EX._info_batch([]) == [] and calls == []
    EX._info_batch([f"File:{i}.jpg" for i in range(85)])
    assert calls == [40, 40, 5]


def test_image_menu_dedupes_captions_numbers_ids_and_limits(monkeypatch):
    titles = [f"File:{i}.jpg" for i in range(30)]
    monkeypatch.setattr(EX.WK, "api", lambda p: {"query": {"pages": [{"images": [{"title": t} for t in titles]}]}} if p.get("prop") == "images"
                        else {"query": {"pages": [{}]}})
    monkeypatch.setattr(EX.WK, "api_commons", lambda p: {"query": {"search": [], "categorymembers": []}} if p.get("list") else {})
    items = [{"title": t, "desc": "Exhibition of cars caption number one" if i < 10 else f"Unique caption {i}"} for i, t in enumerate(titles)]
    monkeypatch.setattr(EX, "_info_batch", lambda ts: [dict(x) for x in items])
    out = EX._image_menu("cars", [{"title": "Car"}], limit=8)
    assert len(out) == 8 and [m["id"] for m in out] == [f"i{i}" for i in range(1, 9)]
    assert sum(1 for m in out if m["desc"].startswith("Exhibition")) == 2


def test_image_menu_cache_is_used_only_when_big_enough(monkeypatch, tmp_path):
    cache = tmp_path / "menu.json"
    calls = []
    monkeypatch.setattr(EX, "_image_menu", lambda t, s, l: (calls.append(1), [{"id": "i1", "title": "t", "desc": "d"}])[1])
    cache.write_text(json.dumps([{"id": f"i{i}"} for i in range(12)]))
    assert len(EX.image_menu("t", [], cache_file=cache)) == 12 and not calls
    cache.write_text(json.dumps([{"id": "i1"}]))                                       # too small: looked up again and re-saved
    assert len(EX.image_menu("t", [], cache_file=cache)) == 1 and calls == [1]
    assert json.loads(cache.read_text())[0]["title"] == "t"
    cache.unlink()
    monkeypatch.setattr(EX, "_image_menu", lambda t, s, l: [])
    assert EX.image_menu("t", [], cache_file=cache) == [] and not cache.exists()      # nothing found: no empty cache file left behind


# ------------------------------------------------------------------------------------------------ fetch_image
def png_bytes(mode="RGBA", size=(40, 30), color=(255, 0, 0, 0)):
    b = io.BytesIO()
    Image.new(mode, size, color).save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def fast(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_fetch_image_puts_transparent_png_on_white(monkeypatch, tmp_path, fast):
    monkeypatch.setattr(EX.requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, content=png_bytes(), raise_for_status=lambda: None))
    f = EX.fetch_image({"title": "File:A b.png", "url": "u"}, tmp_path)
    assert f.name == "file-a-b-png.jpg" and Image.open(f).convert("RGB").getpixel((5, 5))[0] > 240 and Image.open(f).getpixel((5, 5))[1] > 240


def test_fetch_image_cached_file_is_not_downloaded_again(monkeypatch, tmp_path):
    (tmp_path / "file-x-jpg.jpg").write_bytes(b"x")
    monkeypatch.setattr(EX.requests, "get", lambda *a, **k: pytest.fail("downloaded again"))
    assert EX.fetch_image({"title": "File:x.jpg", "url": "u"}, tmp_path).exists()


def test_fetch_image_gives_up_with_none_on_errors_html_and_429(monkeypatch, tmp_path, fast):
    n = {"c": 0}

    def get(*a, **k):
        n["c"] += 1
        return SimpleNamespace(status_code=200, content=b"<html>blocked</html>", raise_for_status=lambda: None)
    monkeypatch.setattr(EX.requests, "get", get)
    assert EX.fetch_image({"title": "File:h.jpg", "url": "u"}, tmp_path, tries=3) is None and n["c"] == 3
    monkeypatch.setattr(EX.requests, "get", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    assert EX.fetch_image({"title": "File:c.jpg", "url": "u"}, tmp_path, tries=2) is None
    monkeypatch.setattr(EX.requests, "get", lambda *a, **k: SimpleNamespace(status_code=429, content=b"", raise_for_status=lambda: None))
    assert EX.fetch_image({"title": "File:r.jpg", "url": "u"}, tmp_path, tries=2) is None
    assert not list(tmp_path.glob("*.jpg"))                                           # no half-written file left to be mistaken for a cached picture


def test_fetch_image_recovers_after_a_429(monkeypatch, tmp_path, fast):
    seq = iter([SimpleNamespace(status_code=429, content=b"", raise_for_status=lambda: None),
                SimpleNamespace(status_code=200, content=png_bytes("RGB", color=(1, 2, 3)), raise_for_status=lambda: None)])
    monkeypatch.setattr(EX.requests, "get", lambda *a, **k: next(seq))
    assert EX.fetch_image({"title": "File:r.jpg", "url": "u"}, tmp_path) is not None


# ------------------------------------------------------------------------------------------------ looks
def test_every_look_is_a_real_theme_and_page():
    for theme, bg in EX.LOOKS:
        assert theme in THEMES and bg in G.PAGES
    assert len({b for _t, b in EX.LOOKS}) == len(EX.LOOKS) == len({t for t, _b in EX.LOOKS})


# ------------------------------------------------------------------------------------------------ build(): look rotation and card fallback (real ffmpeg, tiny fakes)
@pytest.fixture
def mini_build(monkeypatch, tmp_path):
    rendered = []

    class Cfg:
        root = tmp_path

        def get(self, k, d=None):
            return {"tts": {}}.get(k, d)

    class AI:
        def __init__(self, cfg):
            pass

        def available(self):
            return True, "ok"

    def fake_tts(text, cfg, out_dir, previous_text=None, next_text=None, **k):
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / "voice.wav"
        subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=200:duration=0.8", str(wav)], check=True)
        return tts.TTSResult(wav, [{"text": "one", "start": 0.0, "end": 0.3}], 0.8, "fake", "v")

    def fake_render(cfg, spec, W, H, out, theme=None, **k):
        rendered.append({"kind": spec["kind"], "theme": theme, "bg": spec.get("bg"), "spec": spec})
        if spec["kind"] in rendered_fail:
            return None
        return {"file": out}

    rendered_fail: set = set()

    def fake_mp4(src, out, seconds):
        subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=gray:s=1920x1080:r=30:d={seconds:.2f}", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out
    monkeypatch.setattr(EX, "LocalAI", AI)
    monkeypatch.setattr(EX, "sources", lambda topic, titles=None, max_chars=14000: [{"title": "Gear", "url": "https://w/Gear", "text": "x" * 900}])
    monkeypatch.setattr(EX, "image_menu", lambda *a, **k: [])
    monkeypatch.setattr(EX.tts, "synthesize", fake_tts)
    monkeypatch.setattr(EX.GC, "render", fake_render)
    monkeypatch.setattr(EX, "to_mp4", fake_mp4)
    monkeypatch.setattr(EX.MU, "pick", lambda *a, **k: None)
    return Cfg(), rendered, rendered_fail


def script_of(*beats):
    return {"title": "Gears", "beats": [{"section": s, "narration": "one two three four five six seven eight", "visual": v} for s, v in beats]}


def test_build_rotates_looks_per_section_and_sources_card_uses_the_first(mini_build, tmp_path):
    cfg, rendered, _ = mini_build
    sf = tmp_path / "script.json"
    sl = {"kind": "slide", "title": "T", "bullets": ["a", "b"]}
    sf.write_text(json.dumps(script_of(("One", sl), (None, sl), ("Two", sl), ("Three", sl))))
    r = EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)
    got = [(x["kind"], x["theme"], x["bg"]) for x in rendered]
    assert "chapter" not in [g[0] for g in got]                                          # no title cards: the story never stops for a slide
    assert got[0] == ("fullslide", *EX.LOOKS[1]) and got[1] == ("fullslide", *EX.LOOKS[1])      # the opening shots are on the dark blueprint page; after them the look changes with the section
    assert got[2] == ("fullslide", *EX.LOOKS[1]) and got[3] == ("fullslide", *EX.LOOKS[2])
    assert got[-1] == ("fullslide", EX.LOOKS[0][0], EX.LOOKS[0][1])                    # the sources card
    assert Path(r["video"]).exists() and ffmpeg.duration(Path(r["video"])) > 5
    assert "Wikipedia (CC BY-SA 4.0)" in Path(r["credits"]).read_text()


def test_build_more_sections_than_looks_wraps_around(mini_build, tmp_path):
    cfg, rendered, _ = mini_build
    sf = tmp_path / "script.json"
    sl = {"kind": "slide", "title": "T", "bullets": ["a", "b"]}
    sf.write_text(json.dumps(script_of(*[(f"S{i}", sl) for i in range(len(EX.LOOKS) + 1)])))
    EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)
    looks = [(x["theme"], x["bg"]) for x in rendered if x["kind"] == "fullslide"]
    assert looks[:2] == [EX.LOOKS[1]] * 2 and looks[2:len(EX.LOOKS)] == EX.LOOKS[2:] and looks[len(EX.LOOKS)] == EX.LOOKS[0]


def test_build_explicit_theme_overrides_rotation(mini_build, tmp_path):
    cfg, rendered, _ = mini_build
    sf = tmp_path / "script.json"
    sf.write_text(json.dumps(script_of(("One", {"kind": "slide", "title": "T", "bullets": ["a", "b"]}))))
    EX.build(cfg, "gears", theme="slate", script_file=sf, say=lambda *_: None)
    assert {x["theme"] for x in rendered} == {"slate"} and {x["bg"] for x in rendered if x["kind"] != "fullslide" or x["spec"]["title"] != "Sources"} == {None}


def test_build_card_that_will_not_render_falls_back_to_a_title_card(mini_build, tmp_path):
    cfg, rendered, fail = mini_build
    fail.add("bars")
    sf = tmp_path / "script.json"
    sf.write_text(json.dumps(script_of(("One", {"kind": "bars", "title": "Sizes", "labels": ["a", "b"], "values": ["many", "few"]}))))
    EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)
    kinds = [x["kind"] for x in rendered]
    assert kinds.count("bars") == 1 and "chapter" in kinds[1:]
    assert [x for x in rendered if x["kind"] == "chapter"][-1]["spec"]["title"] == "Sizes"


def test_build_stat_card_that_fails_everywhere_raises_a_clear_error(mini_build, tmp_path):
    cfg, rendered, fail = mini_build
    fail.update({"bigstat", "chapter"})
    sf = tmp_path / "script.json"
    sf.write_text(json.dumps(script_of((None, {"kind": "stat", "value": "5", "label": "five"}))))
    with pytest.raises(RuntimeError, match="did not render"):
        EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)


def test_build_photo_that_cannot_be_chosen_uses_a_card(mini_build, tmp_path):
    cfg, rendered, _ = mini_build
    sf = tmp_path / "script.json"
    sf.write_text(json.dumps(script_of(("One", {"kind": "photo", "image": "i99"}))))
    EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)
    assert any(x["kind"] == "fullquote" and "one two three" in x["spec"]["text"] for x in rendered)           # the line's own words, large (never a duplicated, cut-off bullet)


def test_build_without_an_ai_refuses_clearly(mini_build, monkeypatch):
    cfg, _, _ = mini_build
    monkeypatch.setattr(EX, "LocalAI", lambda c: SimpleNamespace(available=lambda: (False, "no key")))
    with pytest.raises(RuntimeError, match="no AI"):
        EX.build(cfg, "gears")


def test_sources_errors_are_clear(monkeypatch):
    monkeypatch.setattr(EX, "find_title", lambda t: None)
    with pytest.raises(ValueError, match="no Wikipedia article"):
        EX.sources("zzzz")
    monkeypatch.setattr(EX.WK, "extract", lambda n, max_chars=0: {"title": n, "text": "short"})
    with pytest.raises(ValueError, match="too little text"):
        EX.sources("x", ["A"])
    monkeypatch.setattr(EX.WK, "extract", lambda n, max_chars=0: None)
    with pytest.raises(ValueError):
        EX.sources("x", ["Missing"])


# ------------------------------------------------------------------------------------------------ wiki.api_commons
def test_api_commons_throttles_and_retries_429(monkeypatch):
    from clipfarm.core import wiki as WK
    import time
    sleeps, seq = [], iter([SimpleNamespace(status_code=429), SimpleNamespace(status_code=429),
                            SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"ok": 1})])
    seen = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(WK.requests, "get", lambda url, **k: (seen.append(url), next(seq))[1])
    assert WK.api_commons({"action": "query"}) == {"ok": 1}
    assert seen == [WK.COMMONS_API] * 3 and 10 in sleeps and 20 in sleeps


def test_api_commons_gives_up_after_five_429s(monkeypatch):
    from clipfarm.core import wiki as WK
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(WK.requests, "get", lambda url, **k: SimpleNamespace(status_code=429))
    with pytest.raises(RuntimeError, match="too many requests"):
        WK.api_commons({})


def test_api_commons_http_error_raises_without_retrying(monkeypatch):
    from clipfarm.core import wiki as WK
    import requests
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    n = {"c": 0}

    def get(url, **k):
        n["c"] += 1

        def boom():
            raise requests.HTTPError("500")
        return SimpleNamespace(status_code=500, raise_for_status=boom)
    monkeypatch.setattr(WK.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        WK.api_commons({})
    assert n["c"] == 1


def test_wiki_api_endpoint_override(monkeypatch):
    from clipfarm.core import wiki as WK
    seen = []
    monkeypatch.setattr(WK.requests, "get", lambda url, **k: (seen.append(url), SimpleNamespace(raise_for_status=lambda: None, json=lambda: {}))[1])
    WK.api({"a": 1})
    WK.api({"a": 1}, WK.COMMONS_API)
    assert seen == [WK.API, WK.COMMONS_API]


# ------------------------------------------------------------------------------------------------ new cards and page styles
SIZES = [(1280, 720), (1080, 1920)]
NEW_THEMES = ["blueprint", "sky", "mint"]
LONG = "Transmission of rotational energy through the continuously variable gearbox assembly"


def scene(kind, spec, theme="clean", size=(1280, 720), secs=8.0):
    return G.build_scene(kind, G.Look(theme, *size), {**spec, "kind": kind, "len": secs}, secs)


def walk(sc, secs=8.0):
    """Every frame we sample is a real image; the card has content at the end."""
    last = None
    for t in (0.0, 0.6, secs * 0.4, secs * 0.7, secs - 0.5):
        last = G.frame_at(sc, t)
        assert last.size == (sc.look.W, sc.look.H)
    return last


@pytest.mark.parametrize("theme", NEW_THEMES + ["clean", "slate", "board", "bold", "glass"])
@pytest.mark.parametrize("bg", list(G.PAGES) + [None, "nonsense"])
def test_page_styles_for_every_theme_return_a_full_image(theme, bg):
    look = G.Look(theme, 640, 360)
    img, ink, dim, dot = G._page(look, bg)
    assert img.size == (640, 360) and len(ink) == 3 and len(dim) == 3 and len(dot) == 3


def test_new_themes_use_their_own_page_colours_and_unknown_pages_equal_dots():
    for t in NEW_THEMES:
        look = G.Look(t, 320, 180)
        assert look.page_c == tuple(THEMES[t]["page"])
        assert max(G._page(look, None)[0].getcolors(10 ** 6))[1] == look.page_c                 # the page colour is the most common one
        assert G._page(look, "nonsense")[0].tobytes() == G._page(look, "dots")[0].tobytes()
    assert G.Look("clean", 320, 180).page_c is None


def test_page_styles_differ_from_each_other():
    look = G.Look("clean", 640, 360)
    imgs = {b: G._page(look, b)[0].tobytes() for b in G.PAGES}
    assert len(set(imgs.values())) == len(G.PAGES)


def test_page_text_contrast_is_readable_for_the_new_themes():
    def lum(c):
        f = lambda v: v / 255 / 12.92 if v / 255 <= 0.03928 else ((v / 255 + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
    for t in NEW_THEMES:
        img, ink, dim, dot = G._page(G.Look(t, 320, 180), "dots")
        bg = img.getpixel((2, 2))
        a, b = sorted([lum(ink), lum(bg)])
        assert (b + 0.05) / (a + 0.05) > 7, t


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("theme,bg", EX.LOOKS)
def test_flow_bars_line_render_in_every_look_and_both_frames(size, theme, bg):
    spec = {"bg": bg, "title": "How it works", "steps": ["Mix", "Heat the mixture", "Cool slowly", "Store"], "labels": ["A", "B", "C"], "values": ["10", "20", "5"],
            "unit": "units", "events": [{"date": "1900", "label": "One"}, {"date": "1950s", "label": "Two"}, {"date": "2000", "label": "Three"}]}
    for kind in ("flow", "bars", "line"):
        sc = scene(kind, spec, theme, size)
        if kind == "flow" and size[1] > size[0]:                                          # portrait: the flow row declines (see test_flow_is_landscape_only...)
            assert sc is None
            continue
        assert sc is not None, kind
        assert walk(sc).getbbox() is not None


@pytest.mark.parametrize("kind,spec", [
    ("flow", {"title": "", "steps": ["a", "b", "c"]}), ("flow", {"title": "T", "steps": ["a", "b"]}), ("flow", {"title": "T", "steps": ["", " ", "c", "d"]}),
    ("flow", {"title": "T"}), ("flow", {"title": "T", "steps": None}), ("flow", {"title": "   ", "steps": ["a", "b", "c"]}),
    ("bars", {"title": "", "labels": ["a", "b"], "values": ["1", "2"]}), ("bars", {"title": "T", "labels": ["a"], "values": ["1"]}),
    ("bars", {"title": "T", "labels": ["a", "b"], "values": ["many", "few"]}), ("bars", {"title": "T", "labels": ["a", "b"], "values": ["0", "0"]}),
    ("bars", {"title": "T", "labels": ["a", "b"], "values": ["-5", "10"]}), ("bars", {"title": "T", "labels": ["a", "b"], "values": ["1"]}),
    ("bars", {"title": "T"}), ("bars", {"title": "T", "labels": ["a", "b"], "values": [None, None]}),
    ("line", {"title": "", "events": [{"date": "1", "label": "a"}] * 3}), ("line", {"title": "T", "events": [{"date": "1", "label": "a"}] * 2}),
    ("line", {"title": "T", "events": ["x", None, {"date": "1"}]}), ("line", {"title": "T"}), ("line", {"title": "T", "events": [{"date": "", "label": "a"}] * 4}),
])
@pytest.mark.parametrize("size", SIZES)
def test_flow_bars_line_with_unusable_input_return_none(kind, spec, size):
    assert scene(kind, spec, "clean", size) is None


@pytest.mark.parametrize("size", SIZES)
def test_flow_with_five_very_long_steps_stays_inside_the_frame(size):
    sc = scene("flow", {"title": "How the whole thing works from start to end", "steps": [LONG] * 5}, "blueprint", size, 10.0)
    if size[1] > size[0]:
        assert sc is None
        return
    img = walk(sc, 10.0)
    assert img.getbbox() is not None and img.getbbox()[0] >= 0 and img.getbbox()[2] <= size[0]


def test_flow_six_steps_are_capped_to_five():
    a = scene("flow", {"title": "T", "steps": list("abcde")}, size=(1280, 720))
    b = scene("flow", {"title": "T", "steps": list("abcdefgh")}, size=(1280, 720))
    assert G.frame_at(a, 7.0).tobytes() == G.frame_at(b, 7.0).tobytes()


def test_flow_times_from_speech_reveal_boxes_when_said():
    sp = {"title": "T", "steps": ["a", "b", "c"], "times": [1.0, 3.0, 5.0]}
    sc = scene("flow", sp, "blueprint", secs=8.0)
    early, mid, late = (G.frame_at(sc, t) for t in (1.5, 3.5, 7.0))
    acc = np.array([0x4f, 0xd1, 0xff])
    cnt = lambda im: int((np.abs(np.asarray(im)[..., :3].astype(int) - acc).sum(axis=2) < 12).sum())  # noqa: E731    accent-coloured pixels: box outlines and arrows
    assert cnt(early) < cnt(mid) < cnt(late)
    short = scene("flow", {**sp, "times": [1.0]}, "blueprint", secs=8.0)                  # a times list of the wrong length is ignored, not an error
    assert short is not None and walk(short).getbbox()


@pytest.mark.parametrize("vals", [["1,800", "900", "450"], ["35%", "12.5%", "0%"], ["$1,200.50", "$300.25", "$10"], ["7.5", "2.25", "1"], ["3 million", "1 million", "2 million"],
                                  ["0", "0", "1"], ["1e3", "5", "2"], ["２０", "5", "2"]])
@pytest.mark.parametrize("size", SIZES)
def test_bars_accept_commas_percent_currency_and_units(vals, size):
    sc = scene("bars", {"title": "Sizes", "labels": ["A", "B", "C"], "values": vals}, "mint", size)
    assert sc is not None and walk(sc).getbbox() is not None


def test_bars_num_parser():
    assert G._num("1,800") == (1800.0, 0, "")
    assert G._num("35%") == (35.0, 0, "%")
    assert G._num("12.50 %") == (12.5, 2, "%")
    assert G._num("$7.5 billion") == (7.5, 1, "billion")
    assert G._num("-4") == (-4.0, 0, "")
    assert G._num("many") is None and G._num("") is None and G._num(None) is None


def test_bars_count_up_and_biggest_is_accent():
    sc = scene("bars", {"title": "T", "labels": ["A", "B"], "values": ["100", "50"]}, "blueprint", secs=8.0)
    early, late = G.frame_at(sc, 1.3), G.frame_at(sc, 6.0)
    assert early.tobytes() != late.tobytes()


@pytest.mark.parametrize("size", SIZES)
def test_bars_big_figures_are_not_cut_off_by_the_frame_edge(size):
    from PIL import ImageDraw
    sc = scene("bars", {"title": "Population", "labels": ["Tokyo", "Delhi"], "values": ["37,400,000", "31,000,000"]}, "blueprint", size)
    rec, orig = [], ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):                                                    # where each piece of text is really drawn (the card's page is always full frame)
        rec.append((text, self.textbbox(xy, text, font=k.get("font"))))
        return orig(self, xy, text, *a, **k)
    ImageDraw.ImageDraw.text = spy
    try:
        G.frame_at(sc, 6.5)
    finally:
        ImageDraw.ImageDraw.text = orig
    figures = [(t, bb) for t, bb in rec if t[0].isdigit()]
    assert figures and all(bb[2] <= size[0] - 8 for _t, bb in figures), figures


def test_bars_values_that_were_not_given_as_strings():
    sc = scene("bars", {"title": "T", "labels": ["A", "B"], "values": [10, 20.5]})
    assert sc is not None and walk(sc).getbbox()


def test_flow_is_landscape_only_portrait_falls_back_to_none():
    """A row of boxes needs a wide frame: in portrait the scene declines (the explainer is 16:9; callers fall back to a slide)."""
    assert scene("flow", {"title": "T", "steps": ["Transmission"] * 5}, "blueprint", (1080, 1920), 10.0) is None
    assert scene("flow", {"title": "T", "steps": ["Transmission"] * 5}, "blueprint", (1280, 720), 10.0) is not None


@pytest.mark.parametrize("size", SIZES)
def test_line_with_long_labels_and_dates_renders(size):
    ev = [{"date": "1939-1945", "label": LONG}] * 5
    sc = scene("line", {"title": "A very long timeline title about the history of everything", "events": ev}, "sky", size, 10.0)
    assert walk(sc, 10.0).getbbox() is not None


def test_line_is_capped_at_five_events_and_uses_speech_times():
    ev = [{"date": str(1900 + i), "label": f"event {i}"} for i in range(8)]
    a = scene("line", {"title": "T", "events": ev}, secs=10.0)
    b = scene("line", {"title": "T", "events": ev[:5]}, secs=10.0)
    assert G.frame_at(a, 9.0).tobytes() == G.frame_at(b, 9.0).tobytes()
    c = scene("line", {"title": "T", "events": ev[:3], "times": [1.0, 4.0, 7.0]}, secs=10.0)
    assert c is not None and walk(c, 10.0).getbbox()


@pytest.mark.parametrize("secs", [0.5, 1.0, 2.0])
def test_new_cards_survive_a_very_short_card(secs):
    for kind, spec in (("flow", {"title": "T", "steps": list("abc")}), ("bars", {"title": "T", "labels": list("ab"), "values": ["1", "2"]}),
                       ("line", {"title": "T", "events": [{"date": "1", "label": "a"}] * 3})):
        sc = scene(kind, spec, "blueprint", secs=secs)
        assert sc is not None
        for t in (0.0, secs / 2, secs - 0.05):
            G.frame_at(sc, t)


@pytest.mark.parametrize("theme", ["clean", "slate", "board", "bold", "mono", "glass", "glass_light"])
@pytest.mark.parametrize("bg", [None, "dots"])
def test_existing_kinds_still_render_without_bg(theme, bg):
    specs = {"fullslide": {"title": "Hello", "bullets": ["one", "two"]}, "bigstat": {"value": "1,939", "label": "things"}, "chapter": {"number": "1", "title": "Part"},
             "timeline": {"title": "T", "events": [{"date": "1", "label": "a"}, {"date": "2", "label": "b"}, {"date": "3", "label": "c"}]},
             "chart": {"title": "C", "labels": ["a", "b"], "values": [1, 2]}}
    for kind, sp in specs.items():
        sc = scene(kind, {**sp, **({"bg": bg} if bg else {})}, theme)
        assert sc is None or walk(sc).size == (1280, 720)


def test_default_page_is_byte_identical_to_dots():
    for theme in ("clean", "slate", "board", "bold", "glass_light"):
        look = G.Look(theme, 640, 360)
        assert G._page(look)[0].tobytes() == G._page(look, "dots")[0].tobytes()


def test_scene_registry_has_the_new_kinds():
    assert {"flow", "bars", "line"} <= set(G.SCENES)


def test_a_spoken_section_title_is_stripped_from_the_narration_and_the_hook_has_no_title(monkeypatch):
    class AI:
        def complete_json(self, task, system, prompt, schema, **kw):
            return type("R", (), {"data": {"title": "T", "beats": [
                {"section": "What it is", "narration": "Section What it is. A QR code is a grid of black and white squares.", "visual": {"kind": "photo"}},
                {"section": "How it works", "narration": "How it works: the scanner finds three big corner squares first.", "visual": {"kind": "photo"}}]}})()
    s = EX.write_script(AI(), "qr", [{"title": "QR", "url": "u", "text": "x" * 900}], 5, [])
    assert s["beats"][0]["narration"].startswith("A QR code is") and s["beats"][0]["section"] == ""
    assert s["beats"][1]["narration"].startswith("the scanner finds") and "How it works" not in s["beats"][1]["narration"]


def test_sound_effects_are_synthesised_placed_on_the_video_timeline_and_never_clip(tmp_path):
    from clipfarm.clipper import soundfx as sfx
    for kind in sfx.KINDS:
        y = sfx.synth(kind, 3)
        assert len(y) > 100 and abs(y).max() > 0.05 and abs(y).max() < 5
    cues = EX.all_cues([{"kind": "animation", "dur": 4.0, "cues": [{"t": 1.0, "kind": "click", "gain": 0.7}]}, {"kind": "stat", "dur": 5.0}, {"kind": "photo", "dur": 3.0}])
    assert [(c["t"], c["kind"]) for c in cues] == [(1.0, "click"), (4.5, "ding")]       # the animation's own cue, then the stat card's ding 0.5 s into its segment
    wav = sfx.render(cues, 10.0, tmp_path / "s.wav")
    import wave
    with wave.open(str(wav)) as w:
        assert w.getnchannels() == 2 and w.getframerate() == 48000 and abs(w.getnframes() / 48000 - 10.0) < 0.05


def test_sound_cues_from_the_ai_are_clamped_to_the_scene_and_unknown_kinds_dropped():
    from clipfarm.clipper import soundfx as sfx

    class AI:
        def complete_json(self, *a, **k):
            return type("R", (), {"data": {"cues": [{"t": 0.5, "kind": "click"}, {"t": 99, "kind": "ding"}, {"t": 1.0, "kind": "laser"}, {"t": -1, "kind": "tap"}, {"t": 2.0, "kind": "whoosh", "gain": 5}]}})()
    out = sfx.cues_for(AI(), "typing", "n", [{"text": "a", "start": 0.1}], 4.0)
    assert [(c["t"], c["kind"]) for c in out] == [(0.5, "click"), (2.0, "whoosh")] and out[1]["gain"] == 1.0


def test_a_hook_animation_is_never_replaced_by_pictures(mini_build, tmp_path):
    cfg, rendered, _ = mini_build
    sf = tmp_path / "script.json"
    sf.write_text(json.dumps(script_of((None, {"kind": "slide", "title": "T", "bullets": ["a", "b"]}))))
    EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)
    assert "montage" not in open(EX.__file__).read()


def test_the_mascot_goes_on_long_segments_only_after_the_hook_and_never_twice_close_together():
    from clipfarm.clipper import mascot as MC
    segs = [{"beat": i, "dur": d, "file": None} for i, d in enumerate([6.0, 9.0, 8.0, 7.5, 12.0, 6.0, 10.0, 9.0, 11.0])] + [{"beat": None, "dur": 5.0, "file": None}]
    got = MC.pick_segments(segs)
    t0, starts = 0.0, []
    for k, s in enumerate(segs):
        if k in got:
            starts.append(t0)
        t0 += s["dur"]
    assert got and all(s >= 6 for s in starts) and all(b - a >= 16 for a, b in zip(starts, starts[1:])) and (len(segs) - 1) not in got


def test_cogsworth_draws_a_frame_that_is_transparent_around_him_and_his_mouth_follows_the_voice(tmp_path):
    from clipfarm.clipper import mascot as MC
    shut, open_ = MC.character(1.0, 0.0, True, 120), MC.character(1.0, 1.0, True, 120)
    assert shut.size == (120, 120) and shut.getpixel((2, 2))[3] == 0 and shut.getpixel((60, 50))[3] == 255
    assert list(shut.getdata()) != list(open_.getdata())


def test_a_photo_that_would_stay_up_too_long_is_drawn_as_an_animation_and_a_short_one_stays_a_photo(mini_build, tmp_path, monkeypatch):
    cfg, rendered, _ = mini_build
    from PIL import Image
    pic = tmp_path / "p.jpg"
    Image.new("RGB", (900, 600), (90, 120, 160)).save(pic)
    monkeypatch.setattr(EX, "image_menu", lambda *a, **k: [{"id": "i1", "desc": "the telescope", "title": "T", "artist": "a", "license": "CC0", "page": "p", "url": "u"}])
    monkeypatch.setattr(EX, "fetch_image", lambda m, c: pic)
    asked = []
    monkeypatch.setattr(EX, "animate", lambda cfg_, ai, tier, concept, *a, **k: asked.append(concept) or None)           # the drawing fails: the photo is the fallback
    long_words = " ".join(["word"] * 40)
    sf = tmp_path / "script.json"
    sf.write_text(json.dumps({"title": "T", "beats": [{"section": "", "narration": "short line here for the first one, nice", "visual": {"kind": "photo", "image": "i1"}},
                                                     {"section": "", "narration": long_words, "visual": {"kind": "photo", "image": "i1"}}]}))
    monkeypatch.setattr(EX.tts, "synthesize", lambda text, *a, **k: type("R", (), {"audio": _silence(tmp_path, 3.0 if len(text.split()) < 20 else 12.0), "words": [], "duration": 3.0 if len(text.split()) < 20 else 12.0,
                                                                                    "provider": "t", "voice": "v"})())
    EX.build(cfg, "gears", script_file=sf, say=lambda *_: None)
    assert len(asked) == 1 and "telescope" in asked[0]


def _silence(tmp_path, seconds):
    import subprocess
    from clipfarm.core import ffmpeg
    p = tmp_path / f"sil{seconds}.mp3"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono", "-t", str(seconds), str(p)], check=True)
    return p


def test_too_many_number_cards_and_slides_become_drawn_animations_and_a_repeated_number_too():
    beats = [{"section": "", "narration": f"line number {i} says something quite interesting here", "visual": {"kind": "stat", "value": "12" if i in (0, 5) else str(i + 100), "label": "x"}} for i in range(7)]
    beats += [{"section": "", "narration": f"slide line {i} says something interesting as well", "visual": {"kind": "slide", "title": "t", "bullets": ["a", "b"]}} for i in range(7)]
    out = EX._clean_script({"title": "T", "beats": beats})["beats"]
    kinds = [b["visual"]["kind"] for b in out]
    assert kinds.count("stat") <= 4 and kinds.count("slide") <= 5 and kinds.count("animation") >= 5
    assert [b["visual"]["value"] for b in out if b["visual"]["kind"] == "stat"].count("12") == 1             # the same number is not shown twice
    assert all("line" in b["visual"]["concept"] for b in out if b["visual"]["kind"] == "animation")
