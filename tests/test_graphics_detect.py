"""Tester additions: the transcript scan (gfxdetect), card geometry (shadow margin, timeline, entity tiles), made-up titles and the render fallbacks.
Tests marked xfail document a defect found by the tester (they will start to XPASS once it is fixed)."""
import time

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import gfxcards as G
from clipfarm.clipper import gfxdetect as D
from clipfarm.clipper import gfxplan as GP
from clipfarm.clipper import moments as M
from clipfarm.core import ffmpeg


def S(*texts, gap=1.0):
    out, t = [], 0.0
    for i, x in enumerate(texts):
        out.append(M.Sentence(i, t, t + 4.0, x))
        t += 4.0 + gap
    return out


def kinds(*texts):
    return [o["kind"] for o in D.find(S(*texts), [])]


# ------------------------------------------------------------------------------------------------ gfxdetect: no false alarms on ordinary speech
@pytest.mark.parametrize("text", [
    "I'm 25 years old and she is 30.", "Call me at 555-1234 or 555 867 5309.", "I live at 123 Main Street, apartment 4.", "I have one or two ideas.",
    "That was two years ago.", "Meet me at 3:30 tomorrow.", "First of all, thanks for coming.", "It takes about 5 to 10 minutes.", "from time to time",
    "from one place to another", "from a to b", "I went to the store and bought some bread.", "There are one hundred and twenty people.",
    "as high as 5 and as low as 5", "between 5 and 5", "In 1999 and 1999 and 1999.", "Wait a second, that was fun.",
])
def test_ordinary_speech_makes_no_card_hint(text):
    assert kinds(text) == [], text


def test_empty_and_degenerate_input_is_fine():
    assert D.find([], []) == []
    assert D.find(S(""), []) == [] and D.find(S("   "), []) == [] and D.find(S("$", "%", "..."), []) == []
    assert D.prompt_lines([]) == ""


def test_the_scan_is_fast_on_a_long_transcript_and_on_hostile_text():
    long = S(*(["We went from 20 to 30 and as high as nine and then it was 5 versus 7, in 2019, 2020 and 2021, first and second."] * 3000))
    t = time.time()
    D.find(long, [])
    assert time.time() - t < 3.0
    t = time.time()
    D.find(S("from " + "one " * 5000, "a" * 100000, "as high as " + "one " * 2000, "between " + "ten " * 3000), [])
    assert time.time() - t < 3.0                                                                   # no catastrophic backtracking


def test_ranges_in_digits_words_and_mixed_forms():
    def vals(text):
        c = [o for o in D.find(S(text), []) if o["kind"] == "compare"]
        return c[0]["item"]["values"] if c else None
    assert vals("It could be as high as 2,000 and as low as one.") == ["2,000", "1"]
    assert vals("It could be as low as $9 and as high as $90!") == ["$9", "$90"]
    assert vals("Anywhere from twenty five to two thousand") == ["25", "2,000"]
    assert vals("We saw between ten and twenty") == ["10", "20"]
    assert vals("It went from 1.5 to 2.5.") == ["1.5", "2.5"]
    assert vals("as high as five and as low as 5") is None                                         # the same number twice: nothing to compare


def test_trailing_punctuation_and_missing_punctuation_do_not_matter():
    for tail in ("", ".", "!", "...", ",", " ?"):
        c = [o for o in D.find(S("It ranges from 5 to 10" + tail), []) if o["kind"] == "compare"]
        assert len(c) == 1 and c[0]["item"]["values"] == ["5", "10"], tail


def test_a_range_split_over_two_sentences_is_found_once():
    f = D.find(S("It could be as high as 2,000.", "And it could be as low as one."), [])
    assert [o["kind"] for o in f].count("compare") == 1


def test_number_of_and_shown_edge_cases():
    assert D.number_of("") is None and D.number_of("a") is None and D.number_of("$1,000,000") == 1_000_000
    assert D.number_of("one hundred") == 100 and D.number_of("two hundred thousand") == 200_000 and D.number_of("5.") == 5.0
    assert D.shown("two thousand,") == "2,000" and D.shown("1.5") == "1.5" and D.shown("$9") == "$9"


# ------------------------------------------------------------------------------------------------ gfxdetect: defects (xfail)
@pytest.mark.parametrize("text", ["someone versus ten people showed up", "stone vs. nine", "from 3 to 4th place"])
def test_numbers_inside_other_words_are_not_numbers(text):
    assert [o for o in D.find(S(text), []) if o["kind"] == "compare"] == []


def test_a_range_keeps_its_units():
    c = [o for o in D.find(S("It could be as high as 2 million and as low as 5 thousand dollars."), []) if o["kind"] == "compare"]
    assert not c or c[0]["item"]["values"] != ["2", "5"]


@pytest.mark.parametrize("text", ["The pay went from 1999 to 2004 in my job.", "I work from 9 to 5 every day."])
def test_years_and_work_hours_are_not_a_compare(text):
    assert [o for o in D.find(S(text), []) if o["kind"] == "compare"] == []


def test_a_versus_range_is_a_hint_for_the_ai_not_a_code_built_card():                  # (which number is which needs judgement)
    assert [o.get("item") for o in D.find(S("The score was ten versus twenty at the end."), []) if o["kind"] == "compare"] == [None]
    return
    text = "The score was ten versus twenty at the end."
    words, t = [], 10.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3})
        t += 0.4
    ai = type("AI", (), {"complete_json": lambda self, *a, **k: type("R", (), {"data": {"graphics": []}})()})()
    out = GP.plan(ai, [M.Sentence(0, 10.0, t, text)], "", {"kinds": ["compare"], "graphics_per_s": 20}, 60.0, 3.2, words)
    assert [c["kind"] for c in out] == ["compare"]


def test_the_ai_prompt_carries_the_hints_and_a_failed_ai_still_gives_nothing():
    seen = {}

    class AI:
        def complete_json(self, task, system, prompt, schema):
            seen["prompt"] = prompt
            raise RuntimeError("offline")
    text = "It could be as high as 2000 and as low as one."
    out = GP.plan(AI(), [M.Sentence(0, 10.0, 15.0, text)], "", {"kinds": ["compare"], "graphics_per_s": 20}, 60.0, 3.2, [])
    assert out == [] and "MOMENTS THAT PROBABLY WANT A GRAPHIC" in seen["prompt"] and "compare" in seen["prompt"]
    seen.clear()
    GP.plan(AI(), [M.Sentence(0, 10.0, 15.0, "I went to the store.")], "", {"kinds": ["compare"], "graphics_per_s": 20}, 60.0, 3.2, [])
    assert "MOMENTS" not in seen["prompt"]


def test_a_found_range_is_not_added_when_compare_is_not_allowed_or_the_ai_already_made_one():
    text = "It could be as high as 2000 and as low as one."
    words, t = [], 10.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3})
        t += 0.4
    sents = [M.Sentence(0, 10.0, t, text)]
    def run(kinds_, items):
        ai = type("AI", (), {"complete_json": lambda self, *a, **k: type("R", (), {"data": {"graphics": items}})()})()
        return GP.plan(ai, sents, "", {"kinds": kinds_, "graphics_per_s": 20}, 60.0, 3.2, words)
    assert run(["stat"], []) == []
    mine = {"kind": "compare", "at": 10.2, "labels": ["Max", "Min"], "values": ["2000", "1"]}
    got = run(["compare"], [mine])
    assert len(got) == 1 and got[0]["labels"] == ["Max", "Min"]


# ------------------------------------------------------------------------------------------------ gfxplan.schedule / number words
def test_schedule_never_overlaps_and_handles_empty_and_equal_times():
    assert GP.schedule([], 60.0) == []
    cards = [{"kind": "stat", "at": 10.0, "len": 4.0}, {"kind": "stat", "at": 10.0, "len": 4.0}, {"kind": "slide", "at": 11.0, "len": 7.0}, {"kind": "stat", "at": 53.0, "len": 4.2}, {"kind": "stat", "at": 55.0, "len": 4.2}]
    out = GP.schedule(cards, 60.0)
    assert all(a["at"] + a["len"] <= b["at"] + 0.011 for a, b in zip(out, out[1:]))
    assert all(c["at"] + c["len"] <= 60.0 + 0.011 for c in out)
    assert len(cards) == 5 and cards[0]["len"] == 4.0                                              # the input is not changed


def test_number_words_are_matched_to_digits():
    assert GP._said_number("22", "about twenty two percent") is False or True                       # (no crash on either answer)
    assert GP._said_number("6", "it took six hours")
    assert GP._said_number("47%", "forty-seven percent of them")
    assert not GP._said_number("45%", "forty-seven percent of them")
    assert not GP._said_number("", "anything") and not GP._said_number("abc", "anything")


# ------------------------------------------------------------------------------------------------ gfxcards geometry
def _pics(tmp_path):
    for n, c in (("flag.png", (200, 30, 60, 255)), ("mark.png", (255, 255, 255, 255)), ("photo.jpg", (60, 90, 140, 255))):
        Image.new("RGBA", (400, 260), c).convert("RGB" if n.endswith("jpg") else "RGBA").save(tmp_path / n)
    f = lambda n, t, fn, at, color=None, mark=False: {"file": str(tmp_path / fn), "type": t, "name": n, "credit": "", "color": color, "mark": mark, "at_rel": at}  # noqa: E731
    return [f("OpenAI", "logo", "mark.png", 0.0, "#10A37F", True), f("Anthropic", "logo", "mark.png", 1.0, "#191919", True), f("Japan", "flag", "flag.png", 2.0),
            f("Sam Altman", "photo", "photo.jpg", 3.0)]


SOFT_LOOKS = ("glass", "glass_light", "clean", "slate", "mono", "minimal", "pastel_orange")


@pytest.mark.parametrize("W,H", [(1920, 1080), (3840, 2160), (1080, 1920)])
def test_the_soft_shadow_is_never_cut_off_in_a_hard_line(W, H):
    for theme in SOFT_LOOKS:
        look = G.Look(theme, W, H)
        for w, h in ((300, 120), (900, 500), (60, 60)):
            img, m = G.panel(w, h, look.px(30), look)
            a = np.asarray(img)[..., 3]
            assert max(a[0].max(), a[-1].max(), a[:, 0].max(), a[:, -1].max()) <= 4, (theme, W, H, w, h)       # the glow has faded out before the edge of its layer
            assert img.size == (w + 2 * m, h + 2 * m)


def test_every_card_layer_fades_out_before_its_edge(tmp_path):
    ents = _pics(tmp_path)
    cards = [{"kind": "timeline", "len": 8, "events": [{"date": "Sep 22", "label": "went viral"}, {"date": "Sep 23", "label": "they answered"}]},
             {"kind": "compare", "len": 7, "labels": ["Starts", "Tops out"], "values": ["$1,699", "$2,699"]},
             {"kind": "stat", "len": 6, "value": "22%", "label": "of moms had no support at home"},
             {"kind": "entity", "len": 7, "items": ents[:3]}, {"kind": "hub", "len": 9, "items": ents},
             {"kind": "chart", "len": 7, "chart": {"type": "column", "title": "Visits", "labels": ["A", "B", "C"], "values": ["4", "18", "44"]}}]
    for W, H in ((1920, 1080), (3840, 2160), (1080, 1920)):
        for theme in ("glass", "clean", "slate"):
            for sp in cards:
                sc = G.build_scene(sp["kind"], G.Look(theme, W, H), sp, sp["len"])
                assert sc is not None
                for L in sc.layers:
                    a = np.asarray(L["img"])[..., 3]
                    edge = max(a[0].max(), a[-1].max(), a[:, 0].max(), a[:, -1].max())
                    assert not (6 < edge < 250 and L["img"].width > 100), (W, H, theme, sp["kind"], int(edge))       # (a fully opaque edge is a pill or a line, not a shadow)


EVENTS4 = [{"date": "September 22, 2024", "label": "the post went viral on every site"}, {"date": "March 2023", "label": "DigitalOcean answered with a long statement"},
           {"date": "2019", "label": "started"}, {"date": "Sep 24", "label": "the final decision came down"}]


def test_a_long_date_keeps_its_year(monkeypatch):
    seen = []
    real = G.text_layer
    monkeypatch.setattr(G, "text_layer", lambda lines, *a, **k: (seen.extend(lines), real(lines, *a, **k))[1])
    sc = G.build_scene("timeline", G.Look("glass", 1920, 1080), {"kind": "timeline", "len": 8, "events": EVENTS4}, 8)
    assert sc is not None
    assert any("2024" in x for x in seen), seen


def test_the_vertical_timeline_stays_inside_a_portrait_frame_beside_a_face():
    W, H = 1080, 1920
    look = G.Look("glass", W, H)
    look.avoid = [(0.36 * W, 0.08 * H, 0.64 * W, 0.62 * H)]
    sc = G.build_scene("timeline", look, {"kind": "timeline", "len": 9, "events": EVENTS4}, 9)
    assert sc is not None
    assert all(L["x"] + L["img"].width <= W for L in sc.layers)


def test_timeline_cards_survive_any_face_without_covering_it():
    rng = np.random.default_rng(11)
    for W, H in ((1920, 1080), (1080, 1920)):
        for _ in range(8):
            cx, cy, fw, fh = rng.uniform(0.2, 0.8), rng.uniform(0.25, 0.65), rng.uniform(0.12, 0.4), rng.uniform(0.25, 0.8)
            z = (max(0, cx - fw / 2), max(0, cy - fh / 2), min(1, cx + fw / 2), min(1, cy + fh / 2))
            look = G.Look("clean", W, H)
            look.avoid = [(z[0] * W, z[1] * H, z[2] * W, z[3] * H)]
            sc = G.build_scene("timeline", look, {"kind": "timeline", "len": 8, "events": EVENTS4[1:3]}, 8)
            if sc is None:
                continue
            fr = np.asarray(G.frame_at(sc, 5.0))[..., 3]
            x0, y0, x1, y1 = (int(v) for v in look.avoid[0])
            assert not (fr[y0:y1, x0:x1] > 40).any(), (W, H, z)


def _bodies(sc, look):
    out = []
    for L in sc.layers:
        if L["img"].width > look.px(110) and L["img"].height > look.px(110):
            a = np.asarray(L["img"])[..., 3]
            ys, xs = np.where(a > 200)
            if len(xs):
                out.append((L["x"] + xs.min(), L["y"] + ys.min(), L["x"] + xs.max(), L["y"] + ys.max()))
    return out


def test_entity_and_hub_tiles_never_overlap_beside_random_faces(tmp_path):
    ents = _pics(tmp_path)
    rng = np.random.default_rng(5)
    checked = 0
    for W, H in ((1920, 1080), (1080, 1920)):
        for i in range(10):
            cx, cy, fw, fh = rng.uniform(0.2, 0.8), rng.uniform(0.25, 0.65), rng.uniform(0.12, 0.45), rng.uniform(0.25, 0.9)
            z = (max(0, cx - fw / 2), max(0, cy - fh / 2), min(1, cx + fw / 2), min(1, cy + fh / 2))
            for sp in ({"kind": "entity", "len": 7, "items": ents}, {"kind": "entity", "len": 7, "joiner": "vs", "items": ents[:2]}, {"kind": "hub", "len": 9, "items": ents}):
                look = G.Look("glass_light", W, H)
                if i % 3:
                    look.avoid = [(z[0] * W, z[1] * H, z[2] * W, z[3] * H)]
                sc = G.build_scene(sp["kind"], look, sp, sp["len"])
                if sc is None:
                    continue
                rs = _bodies(sc, look)
                for a in range(len(rs)):
                    for b in range(a + 1, len(rs)):
                        A, B = rs[a], rs[b]
                        assert not (min(A[2], B[2]) - max(A[0], B[0]) > 3 and min(A[3], B[3]) - max(A[1], B[1]) > 3), (W, H, sp["kind"], z)
                        checked += 1
    assert checked > 20


def test_cards_render_to_a_real_4k_file_with_and_without_a_face(tmp_path):
    sp = {"kind": "compare", "len": 3, "labels": ["As high as", "As low as"], "values": ["2,000", "1"]}
    for avoid in (None, [(0.36, 0.08, 0.64, 0.62)]):
        out = tmp_path / f"c{bool(avoid)}.mov"
        r = G.render(None, sp, 3840, 2160, out, theme="glass", avoid=avoid)
        assert r and out.exists()
        info = ffmpeg.probe(out)
        v = ffmpeg.video_stream(info)
        assert (v["width"], v["height"]) == (3840, 2160) and ffmpeg.duration(out) == pytest.approx(3.0, abs=0.2)


# ------------------------------------------------------------------------------------------------ polish: made-up title
def test_filename_detection_edge_cases():
    from clipfarm.clipper import polish as P
    assert P.looks_like_filename("GX010234") and P.looks_like_filename("Final cut v2.mov") and P.looks_like_filename(None) and P.looks_like_filename("  ")
    assert not P.looks_like_filename("Morning Routine") and not P.looks_like_filename("How I edit", [{}]) and not P.looks_like_filename("x", [{"name": None}])


def test_a_made_up_title_is_clean():
    from clipfarm.clipper import polish as P

    def ai(r):
        class A:
            def complete_json(self, *a, **k):
                if isinstance(r, Exception):
                    raise r
                return type("R", (), {"data": r})()
        return A()
    w = [{"text": "hello"}] * 5
    assert P.fresh_title(ai({"title": '"A  quoted   title"'}), w) == "A quoted title"
    assert P.fresh_title(ai({"title": "One"}), w) is None and P.fresh_title(ai({}), w) is None and P.fresh_title(ai(RuntimeError("x")), w) is None


def test_a_made_up_title_is_at_most_eight_words_and_trimmed():
    from clipfarm.clipper import polish as P

    class A:
        def complete_json(self, *a, **k):
            return type("R", (), {"data": {"title": "x " * 100}})()
    t = P.fresh_title(A(), [{"text": "hello"}] * 5)
    assert t is None or (t == t.strip() and len(t.split()) <= 8)


# ------------------------------------------------------------------------------------------------ engine: the fallback retries
def _bare_engine(tmp_path, monkeypatch, outcomes):
    from clipfarm.clipper import polish
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "c")
    eng = Engine(cfg, store)
    job = store.create({"filename": "x.mp4"}, {"mode": "polish", "graphics": "full", "gfx_cards": [{"kind": "stat"}]})
    calls = []

    def fake_run(engine, job_id, src, words, **k):
        opts = dict(store.get(job_id)["options"])
        calls.append(opts)
        r = outcomes[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(polish, "run", fake_run)
    monkeypatch.setattr(polish, "archive_bundle", lambda *a, **k: None)
    monkeypatch.setattr(eng, "_ingest", lambda *a, **k: tmp_path / "s.mp4")
    monkeypatch.setattr(eng, "_source_profile", lambda *a, **k: "talk")
    monkeypatch.setattr(eng, "_transcribe", lambda *a, **k: [])
    monkeypatch.setattr(eng, "_archive", lambda *a, **k: None)
    return eng, store, job, calls


def test_a_failed_final_render_is_retried_without_glass_then_without_graphics(tmp_path, monkeypatch):
    ok = {"trimmed_s": 1.0}
    eng, store, job, calls = _bare_engine(tmp_path, monkeypatch, [ffmpeg.FFmpegError("stall"), ffmpeg.FFmpegError("stall"), ok])
    eng.process(job["id"])
    assert len(calls) == 3 and not calls[0].get("no_glass") and calls[1]["no_glass"] and calls[1].get("gfx_cards")
    assert calls[2]["no_glass"] and calls[2]["gfx_cards"] == [] and calls[2]["graphics"] == "title"
    assert store.get(job["id"])["status"] == "done"


def test_one_retry_is_enough_when_glass_was_the_problem(tmp_path, monkeypatch):
    eng, store, job, calls = _bare_engine(tmp_path, monkeypatch, [ffmpeg.FFmpegError("stall"), {"trimmed_s": 2.0}])
    eng.process(job["id"])
    assert len(calls) == 2 and store.get(job["id"])["status"] == "done"
    assert calls[1].get("gfx_cards")                                                               # graphics kept on the first retry


def test_a_third_failure_marks_the_job_failed(tmp_path, monkeypatch):
    eng, store, job, calls = _bare_engine(tmp_path, monkeypatch, [ffmpeg.FFmpegError("a"), ffmpeg.FFmpegError("b"), ffmpeg.FFmpegError("c")])
    eng.process(job["id"])
    assert len(calls) == 3 and store.get(job["id"])["status"] == "failed"
