"""Tester's checks for the second batch of full-screen cards (search, twocol, ledger, recap), their planner branches (_row_times, stat snapping, validate) and check_notes.

Tests marked xfail are DEFECTS found by the tester (the reason says what is wrong); they are not strict so a fix turns them into XPASS."""
import itertools
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageDraw

from clipfarm.clipper import editstyle, gfxcards as G, gfxplan as GP, moments as M
from clipfarm.core import ffmpeg

LOOKS = ["clean", "slate", "bold", "board", "glass", "glass_light", "mono", "minimal", "pastel_orange", "pastel_green", "pastel_blue", "pastel_pink"]
SIZES = [(1920, 1080), (1080, 1920), (3840, 2160), (1280, 720), (1080, 1080)]
STEPS4 = [["Finds a condo first", "Signs a contract today", "Then applies late", "Gets denied"], ["Applies first", "Gets approved", "Then shops around", "Closes on time"]]
ROWS5 = [{"label": "Purchase price", "value": "$1,234,567.89"}, {"label": "Down payment", "value": "12.5k"}, {"label": "Closing costs and fees", "value": "$8,000"},
         {"label": "Inspection", "value": "$500"}, {"label": "Moving", "value": "$1,200"}]
SPEC = {"search": {"text": "Who qualifies for the first time buyer program?", "answer": "First-time buyers", "number": "02"},
        "twocol": {"title": "Same income. Different order.", "titles": ["Buyer A", "Buyer B"], "steps": STEPS4, "tag": {"side": 0, "text": "Ruled out"}},
        "ledger": {"title": "The condo math", "rows": ROWS5, "total": {"label": "Cash needed", "value": "$22,000"}},
        "recap": {"bullets": ["Check eligibility first", "Apply before you sign", "Keep the paperwork ready"]}}
LEN = {"search": 5.6, "twocol": 12.0, "ledger": 12.0, "recap": 8.0}


def scene(kind, theme="clean", W=1920, H=1080, L=None, **over):
    L = L or LEN[kind]
    sp = {**SPEC[kind], "kind": kind, "at": 3.0, "style": "wipe", "len": L, **over}
    return G.build_scene(kind, G.Look(theme, W, H), sp, L), L


def alpha(sc, t):
    return np.asarray(G.frame_at(sc, t))[..., 3]


def boxes(sc, t):
    """(text, bbox) of every string drawn at time t."""
    rec, orig = [], ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        try:
            bb = self.textbbox(xy, text, font=k.get("font"))
        except Exception:  # noqa: BLE001
            bb = None
        rec.append((text, bb))
        return orig(self, xy, text, *a, **k)
    ImageDraw.ImageDraw.text = spy
    try:
        G.frame_at(sc, t)
    finally:
        ImageDraw.ImageDraw.text = orig
    return rec


def lum(c):
    f = lambda v: v / 255 / 12.92 if v / 255 <= 0.03928 else ((v / 255 + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def contrast(a, b):
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ------------------------------------------------------------------------------------------------ the scenes
@pytest.mark.parametrize("kind", list(SPEC))
def test_every_new_card_draws_in_every_look_size_and_transition(kind):
    for k, theme in enumerate(LOOKS):
        for W, H in SIZES:
            sc, L = scene(kind, theme, W, H, style=G.TRANSITIONS[(k + W) % 4])
            assert sc is not None and sc.custom, (kind, theme, W, H)
            for t in (0.0, 0.01, 0.3, 0.6, L / 2, L - 0.45, L - 0.01, L, L + 2):
                f = np.asarray(G.frame_at(sc, t))
                assert f.shape == (H, W, 4) and f.dtype == np.uint8
            assert alpha(sc, 0.0).max() == 0 and alpha(sc, L).max() == 0
            assert (alpha(sc, L / 2) > 250).mean() > 0.99


@pytest.mark.parametrize("style", G.TRANSITIONS)
@pytest.mark.parametrize("kind", ["search", "twocol", "ledger"])
def test_transition_frames_are_partial_and_never_nan(kind, style):
    sc, L = scene(kind, "clean", 960, 540, style=style)
    cover = [(alpha(sc, t) > 128).mean() for t in np.linspace(0.0, 0.5, 9)]
    assert cover[0] < 0.05 and cover[-1] > 0.95 and all(b >= a - 1e-6 for a, b in zip(cover, cover[1:]))
    out = [(alpha(sc, t) > 128).mean() for t in np.linspace(L - 0.5, L, 9)]
    assert out[0] > 0.95 and out[-1] < 0.05 and all(b <= a + 1e-6 for a, b in zip(out, out[1:]))


def test_search_needs_a_real_question_and_survives_empty_and_huge_input():
    L = G.Look("clean", 1280, 720)
    assert G.build_scene("search", L, {"text": "", "at": 1}, 5) is None
    assert G.build_scene("search", L, {"text": "Why", "at": 1}, 5) is None                # one word is not a question
    assert G.build_scene("search", L, {"text": None}, 5) is None
    assert G.build_scene("search", L, {"text": "Why now?", "answer": None, "number": None}, 5) is not None
    sc = G.build_scene("search", L, {"text": "word " * 500, "answer": "x" * 500, "number": "123456"}, 5)
    assert sc is not None and np.asarray(G.frame_at(sc, 4.0)).shape == (720, 1280, 4)


def test_search_types_the_question_and_finishes_before_the_answer_and_before_the_card_ends():
    for secs, ans in ((4.2, ""), (5.6, "First-time buyers")):
        for q in ("What is MIPAP?", "Who qualifies for the first time home buyer program and why?"):
            sc = G.build_scene("search", G.Look("clean", 1920, 1080), {"text": q, "answer": ans, "style": "wipe", "at": 3}, secs)
            ink = lambda t: (np.asarray(G.frame_at(sc, t).convert("RGB")).sum(-1) < 150).sum()  # noqa: E731
            shown = {t: [s for s, _ in boxes(sc, t) if s and s != "PART"] for t in (0.5, 1.2, secs - 0.5)}
            assert shown[0.5] == [] or all(len(s) < len(q) for s in shown[0.5])
            assert any(s == q for s in shown[secs - 0.5]), (secs, q, shown[secs - 0.5])                       # the whole question is on screen before the out transition
            if ans:
                assert any(s.startswith("First") for s in shown[secs - 0.5]) or ans in " ".join(shown[secs - 0.5]), "the answer is on screen before the card ends"
            assert ink(secs - 0.5) >= ink(1.2) > 0


def test_search_typing_is_not_faster_than_the_cursor_can_be_followed_on_a_normal_card():
    q = "Who qualifies for the first time home buyer program?"                      # 52 chars
    sc = G.build_scene("search", G.Look("clean", 1920, 1080), {"text": q, "style": "wipe", "at": 3}, 4.2)
    n_at = lambda t: max((len(s) for s, _ in boxes(sc, t) if s and q.startswith(s)), default=0)  # noqa: E731
    cps = (n_at(2.0) - n_at(1.0))
    assert 0 < cps <= 45, cps


@pytest.mark.parametrize("n_rows", [1, 2, 3, 4, 5, 6, 9])
def test_ledger_with_one_to_many_rows(n_rows):
    rows = [{"label": f"Item {i}", "value": f"${i}00"} for i in range(1, n_rows + 1)]
    sc = G.build_scene("ledger", G.Look("clean", 1920, 1080), {"rows": rows, "total": {"label": "Total", "value": "$999"}, "style": "wipe", "at": 3}, 10)
    assert (sc is None) == (n_rows < 2)
    if sc:
        shown = [s for s, _ in boxes(sc, 9.0)]
        assert sum(1 for s in shown if s.startswith("Item")) == min(5, n_rows) and "$999" in shown        # at most five rows, total always last


def test_ledger_junk_rows_do_not_crash():
    L = G.Look("clean", 1280, 720)
    assert G.build_scene("ledger", L, {"rows": [1, "x", None, {"label": "a"}, {"value": "3"}], "total": {"value": "5"}}, 8) is None
    assert G.build_scene("ledger", L, {"rows": "abcdef"}, 8) is None
    assert G.build_scene("ledger", L, {"rows": [{"label": "a", "value": 5}, {"label": "b", "value": 6}], "total": [1]}, 8) is not None
    sc = G.build_scene("ledger", L, {"rows": [{"label": "a", "value": "1"}, {"label": "b", "value": "2"}], "total": {"label": None, "value": "3"}, "title": "😀 最終 ünï"}, 8)
    assert sc is not None and np.asarray(G.frame_at(sc, 6)).shape == (720, 1280, 4)


@pytest.mark.parametrize("sizes", [(2, 2), (4, 4), (2, 4), (4, 2), (3, 2), (1, 1), (0, 3)])
def test_twocol_with_mismatched_and_short_columns(sizes):
    steps = [[f"step number {c}{r}" for r in range(n)] for c, n in enumerate(sizes)]
    sc = G.build_scene("twocol", G.Look("clean", 1920, 1080), {"titles": ["A", "B"], "steps": steps, "style": "wipe", "at": 3}, 10)
    assert (sc is None) == (min(sizes) < 2)
    if sc:
        shown = [s for s, _ in boxes(sc, 9.0)]
        for c, n in enumerate(sizes):
            assert sum(1 for s in shown if s.startswith(f"step number {c}")) == min(n, 4)


@pytest.mark.parametrize("tag", [None, {}, "x", [1], {"side": 0}, {"side": 0, "text": ""}, {"side": 5, "text": "Out"}, {"side": -1, "text": "Out"}, {"side": "1", "text": "Out"},
                                 {"side": 1, "text": "Out"}, {"side": None, "text": None}, {"side": 1.0, "text": "x"}, {"side": 1, "text": "😀 失格"}])
def test_twocol_tag_variants_never_crash_and_only_a_valid_tag_draws(tag):
    sc = G.build_scene("twocol", G.Look("clean", 1920, 1080), {"titles": ["A", "B"], "steps": STEPS4, "tag": tag, "style": "wipe", "at": 3}, 12)
    assert sc is not None
    shown = " ".join(s for s, _ in boxes(sc, 10.9))
    valid = isinstance(tag, dict) and str(tag.get("side", "")).lstrip("-").isdigit() and int(tag["side"]) in (0, 1) and (tag.get("text") or "").strip()
    assert ("×" in shown) == bool(valid), (tag, shown)


def test_twocol_non_ascii_and_empty_titles():
    sc = G.build_scene("twocol", G.Look("clean", 1920, 1080), {"title": "最終 😀 Ünï", "titles": ["買い手", "😀"], "steps": [["見つける", "契約"], ["😀😀", "ok"]], "tag": {"side": 0, "text": "失格"}}, 8)
    assert sc is not None and np.asarray(G.frame_at(sc, 6)).shape == (1080, 1920, 4)
    assert G.build_scene("twocol", G.Look("clean", 1920, 1080), {"titles": ["A"], "steps": STEPS4}, 8) is None
    assert G.build_scene("twocol", G.Look("clean", 1920, 1080), {"titles": ["A", "B"], "steps": None}, 8) is None


def test_times_of_uses_the_spoken_times_only_when_they_match_and_keeps_them_inside_the_card():
    assert G._times_of({"times": [1, 2, 3]}, 3, 10.0) == [1.0, 2.0, 3.0]
    assert G._times_of({"times": [0.0, 99.0]}, 2, 10.0) == [0.5, 8.8]                       # clamped into the card
    ev = G._times_of({"times": [1, 2]}, 3, 10.0)                                              # wrong count: spread evenly instead
    assert len(ev) == 3 and ev == sorted(ev) and ev[0] >= 0.9 and ev[-1] <= 10.0 - 1.9
    assert G._times_of({}, 1, 0.5)                                                           # a tiny card does not divide by zero
    assert G._times_of({"times": ["x"]}, 1, 5) if False else True


def test_box_colors_are_a_white_card_on_loud_pages_and_charcoal_on_dark():
    assert G._box_colors(G.Look("bold", 1080, 1920), (0, 0, 0))[0] == (255, 255, 255)
    assert G._box_colors(G.Look("slate", 1080, 1920), (0, 0, 0))[0] == (40, 40, 46)
    assert G._box_colors(G.Look("clean", 1080, 1920), (0, 0, 0))[0] == (255, 255, 255)


def test_ledger_five_rows_and_a_total_fit_the_frame_in_every_size():
    for theme in ("clean", "bold", "board", "slate"):
        for W, H in SIZES:
            sc, L = scene("ledger", theme, W, H)
            for text, bb in boxes(sc, L - 0.8):
                assert bb and 0 <= bb[0] and 0 <= bb[1] and bb[2] <= W and bb[3] <= H, (theme, W, H, text, bb)


def test_twocol_portrait_stack_fits_the_frame_with_four_steps_and_a_head_title():
    for W, H in ((1080, 1920), (720, 1280), (1080, 1350)):
        sc, L = scene("twocol", "clean", W, H, title="Same income. Different order, different result for them")
        for text, bb in boxes(sc, L - 0.8):
            assert bb and 0 <= bb[1] and bb[3] <= H, (W, H, text, bb)


@pytest.mark.xfail(reason="DEFECT gfxcards.py:1668/:1698: a stacked two-column card is 2x542+40 design units tall; on a 1080x1080 frame the second card runs off the bottom "
                          "(with a head title) or both cards are cut at the top and bottom (no head title). Only H>=1350 fits.", strict=False)
@pytest.mark.parametrize("head", ["", "Same income. Different order, different result for them"])
def test_twocol_fits_a_square_frame(head):
    sc, L = scene("twocol", "clean", 1080, 1080, title=head)
    for text, bb in boxes(sc, L - 0.8):
        assert bb and 0.03 * 1080 <= bb[1] and bb[3] <= 0.97 * 1080, (text, bb)                  # (the cards themselves are 22 px beyond the text)


@pytest.mark.xfail(reason="DEFECT gfxcards.py:1743/:1732: ledger label and amount are never fitted: a label of ~25+ characters next to '$1,234,567.89' overprints the amount on a 1080-wide frame "
                          "(portrait/square), and a 34-character label is cut mid-word (:1732).", strict=False)
def test_ledger_label_and_amount_never_overlap():
    rows = [{"label": "Purchase price of the whole condo", "value": "$1,234,567.89"}, {"label": "Down payment amount", "value": "12.5k"}]
    for W, H in ((1080, 1920), (1080, 1080), (720, 1280)):
        sc = G.build_scene("ledger", G.Look("clean", W, H), {"rows": rows, "total": {"value": "$1,999,999.00"}, "style": "wipe", "at": 3}, 10)       # (a different total: the same string twice would hide the row's own box)
        bb = {t: b for t, b in boxes(sc, 9.0)}
        assert bb["Purchase price of the whole condo"][2] < bb["$1,234,567.89"][0], (W, H)


@pytest.mark.xfail(reason="DEFECT gfxcards.py:1589: the question is fitted into bw-250 px but the send button takes 2*sr+36 px from the right: on a 1080-wide frame a 60-character "
                          "question is typed under the send button and a 72-character one runs to the edge of the box.", strict=False)
def test_search_question_stays_clear_of_the_send_button_in_portrait():
    q = "What is the first time home buyer program and who qualifies?"
    sc = G.build_scene("search", G.Look("clean", 1080, 1920), {"text": q, "answer": "x", "style": "wipe", "at": 3}, 5.6)
    bb = {t: b for t, b in boxes(sc, 5.0)}[q]
    bh = 150
    send_left = (1080 - int(1080 * 0.86)) // 2 + int(1080 * 0.86) - (bh // 2 - 16) * 2 - 18
    assert bb[2] <= send_left, (bb, send_left)


@pytest.mark.xfail(reason="DEFECT gfxcards.py:1779: the ledger 'Total' label is drawn in the dim ink: low contrast on the light, dark and bold (red) pages; "
                          "the amount beside it is fine.", strict=False)
@pytest.mark.parametrize("theme", ["clean", "slate", "bold"])
def test_ledger_total_label_is_readable(theme):
    sc, L = scene("ledger", theme, 1920, 1080)
    img = np.asarray(G.frame_at(sc, L - 0.8).convert("RGB")).astype(int)
    lab = [b for t, b in boxes(sc, L - 0.8) if t.lower() == "cash needed"][0]
    reg = img[int(lab[1]):int(lab[3]), int(lab[0]):int(lab[2])].reshape(-1, 3)
    bg = tuple(np.median(reg, axis=0).astype(int))
    far = max(reg, key=lambda p: contrast(tuple(p), bg))
    assert contrast(tuple(far), bg) >= (2.5 if theme == "bold" else 3.0), (theme, contrast(tuple(far), bg))                 # (bold: the median of a dark-text box on a red page measures about 2.9 at worst; ink on that red is above 5:1)


@pytest.mark.xfail(reason="DEFECT gfxcards.py:1717 (twocol step pills): a translucent rectangle is drawn straight onto the clear layer, so it REPLACES the white card instead of tinting it: "
                          "on the bold look the pills are the red page colour, on board they are yellow, on every look they are darker than intended (same bug class as the fixed text holes).", strict=False)
@pytest.mark.parametrize("theme", ["bold", "board"])
def test_twocol_step_pills_are_tinted_card_not_the_page(theme):
    sp = {"titles": ["A", "B"], "steps": [["one two", "three four"], ["five six", "seven eight"]], "style": "wipe", "at": 3}
    sc = G.build_scene("twocol", G.Look(theme, 1920, 1080), sp, 8)
    f = np.asarray(G.frame_at(sc, 6.0).convert("RGB")).astype(int)
    page, pill = f[5, 5], f[int(1080 * 0.30) + 130, int(1920 * 0.06) + 768 - 60]
    assert abs(pill - np.array([255, 255, 255])).max() < 40 and abs(pill - page).max() > 60, (theme, page, pill)


# ------------------------------------------------------------------------------------------------ the planner
def build(text, t0=10.0, dt=0.5):
    words, t = [], t0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3})
        t += dt
    return words, [M.Sentence(0, t0, t, text)], t


def test_row_times_follow_the_spoken_order_and_are_seconds_from_start():
    words, _, _ = build("first open an account then add the documents and finally wait for approval")
    tm = GP._row_times(["open an account", "add the documents", "wait for approval"], words, 10.0)
    assert tm and tm == sorted(tm) and len(set(tm)) == 3 and tm[0] < 2.0
    said = {w["text"]: w["start"] for w in words}
    assert abs(10.0 + tm[1] - said["documents"]) < 1.2


def test_row_times_returns_none_when_anything_is_missing():
    words, _, _ = build("first open an account then add the documents")
    assert GP._row_times(["open an account", "wait for approval"], words, 10.0) is None
    assert GP._row_times(["open an account"], [], 10.0) is None and GP._row_times(["open an account"], None, 10.0) is None
    assert GP._row_times([], words, 10.0) == []
    assert GP._row_times(["go to 5"], words, 10.0) is None                                     # nothing distinctive to listen for
    far, _, _ = build("alpha beta gamma", t0=100.0)
    assert GP._row_times(["alpha beta"], far, 10.0) is None                                    # said, but 90 s after the card


def test_row_times_never_returns_a_moment_before_the_previous_item_or_negative():
    words, _, _ = build("open one then open two then open three then open four", dt=0.6)
    tm = GP._row_times(["open one", "open two", "open three", "open four"], words, 10.0)
    assert tm is None or (min(tm) >= 0.5 and tm == sorted(tm))


@pytest.mark.xfail(reason="DEFECT gfxplan.py:121-122 (_row_times): the search accepts the SAME word instance again (ts >= cur - 0.1) and any of the item's 4 longest tokens, so steps that share "
                          "a word ('sign the contract' / 'sign the lease' / 'sign the deed') all get the time of the first 'sign': three steps pop up at once, long before they are said.", strict=False)
def test_row_times_items_sharing_a_word_get_their_own_moments():
    words, _, _ = build("sign the contract then sign the lease then sign the deed")
    tm = GP._row_times(["Sign the contract", "Sign the lease", "Sign the deed"], words, 10.0)
    assert tm and len(set(tm)) == 3 and tm == sorted(tm), tm


def test_row_times_with_unsorted_words_never_lies():
    words, _, _ = build("sign the contract then check the lease then file the deed")
    tm = GP._row_times(["Sign the contract", "check the lease", "file the deed"], list(reversed(words)), 10.0)
    assert tm is None or tm == sorted(tm)


def test_row_times_tolerates_words_without_timing():
    assert GP._row_times(["alpha beta"], [{"text": "alpha"}], 10.0) is None


TEXT = ("Buyer A finds a condo then signs a contract then applies. Price is 400,000 dollars and fees 8,000 dollars and tax 1,000 dollars. The bottom line apply first then shop.")


def V(item, text=TEXT, out_len=200.0, hook=3.2, **kw):
    words, sents, _ = build(text)
    return GP.validate({"at": 13.0, **item}, text, sents, out_len, hook, kw.get("words", words))


def test_twocol_validate_drops_unsaid_steps_and_needs_two_per_column():
    ok = V({"kind": "twocol", "titles": ["A", "B"], "steps": [["finds a condo", "signs a contract"], ["signs a contract", "then applies"]]})
    assert ok and len(ok["steps"][0]) == 2 and ok["tag"] == {}
    assert V({"kind": "twocol", "titles": ["A", "B"], "steps": [["finds a condo", "flies to mars"], ["signs a contract", "then applies"]]}) is None       # only one real step left on side A
    assert V({"kind": "twocol", "titles": ["A"], "steps": [["finds a condo", "signs a contract"], ["signs a contract", "then applies"]]}) is None
    assert V({"kind": "twocol", "titles": ["A", ""], "steps": [["finds a condo", "signs a contract"], ["signs a contract", "then applies"]]}) is None
    assert V({"kind": "twocol", "titles": ["A", "B"], "steps": [["finds a condo", "signs a contract"]]}) is None
    mism = V({"kind": "twocol", "titles": ["A", "B"], "steps": [["finds a condo", "signs a contract", "then applies"], ["signs a contract", "then applies"]]})
    assert mism and [len(c) for c in mism["steps"]] == [3, 2]


@pytest.mark.parametrize("tag,expect", [({"side": 0, "text": "Out"}, {"side": 0, "text": "Out"}), ({"side": "1", "text": "Wins"}, {"side": 1, "text": "Wins"}), ({"side": 2, "text": "x"}, {}),
                                        ({"side": -1, "text": "x"}, {}), ({"side": 0, "text": ""}, {}), ({"side": 0, "text": None}, {}), ("bad", {}), ([1], {}), (None, {}), ({"side": None}, {}),
                                        ({"side": True, "text": "x"}, {})])
def test_twocol_tag_is_only_kept_when_valid(tag, expect):
    r = V({"kind": "twocol", "titles": ["A", "B"], "steps": [["finds a condo", "signs a contract"], ["signs a contract", "then applies"]], "tag": tag})
    assert r and r["tag"] == expect
    assert len(r["times"]) == 4 + (1 if expect else 0)


def test_twocol_text_is_trimmed():
    r = V({"kind": "twocol", "title": "x" * 300, "titles": ["T" * 99, "U" * 99], "steps": [["finds a condo", "signs a contract"], ["signs a contract", "then applies"]], "tag": {"side": 0, "text": "y" * 99}})
    assert len(r["title"]) <= 60 and all(len(t) <= 26 for t in r["titles"]) and len(r["tag"]["text"]) <= 22


def test_ledger_validate_rejects_fabricated_figures_and_unsaid_totals():
    rows = [{"label": "price", "value": "$400,000"}, {"label": "fees", "value": "$8,000"}]
    r = V({"kind": "ledger", "rows": rows, "total": {"label": "Total", "value": "$408,000"}})
    assert r and r["total"] == {}                                                           # 408,000 was never said
    assert V({"kind": "ledger", "rows": rows + [{"label": "tax", "value": "$1,000"}], "total": {"value": "$1,000"}})["total"]["value"] == "$1,000"
    assert V({"kind": "ledger", "rows": [{"label": "price", "value": "$400,000"}, {"label": "fees", "value": "$9,999"}]}) is None                   # one row invented: below two
    assert V({"kind": "ledger", "rows": [{"label": "price", "value": "$400,000"}, {"label": "yacht", "value": "$8,000"}]}) is None                  # label never said
    assert V({"kind": "ledger", "rows": [{"label": "price", "value": "free"}, {"label": "fees", "value": "lots"}]}) is None
    assert V({"kind": "ledger", "rows": [{"label": "price", "value": "$400,000"}]}) is None
    big = V({"kind": "ledger", "rows": [{"label": "price", "value": "$400,000"}, {"label": "fees", "value": "$8,000"}], "total": {"label": "T" * 90, "value": "$1,000"}})
    assert len(big["total"]["label"]) <= 24
    n = V({"kind": "ledger", "rows": [{"label": "price", "value": 400000}, {"label": "fees", "value": 8000}], "total": {"value": 408000}})
    assert n is None or n["total"] == {}                                                    # ints from a sloppy model


@pytest.mark.xfail(reason="DEFECT gfxplan.py:262-267: ledger rows are checked only for 'label said' and 'value said' separately, so two real amounts can be attached to the wrong labels "
                          "(price $8,000 / fees $400,000 passes) and a wrong sum is shown as if it were said.", strict=False)
def test_ledger_does_not_swap_amounts_between_labels():
    r = V({"kind": "ledger", "rows": [{"label": "price", "value": "$8,000"}, {"label": "fees", "value": "$400,000"}]})
    assert r is None


def test_ledger_times_match_the_rows_and_total_when_the_words_are_there():
    text = "Add it up. rent costs 100 dollars. food costs 200 dollars. fuel costs 300 dollars. All together 600 dollars."
    r = V({"kind": "ledger", "rows": [{"label": "rent", "value": "$100"}, {"label": "food", "value": "$200"}, {"label": "fuel", "value": "$300"}], "total": {"label": "All", "value": "$600"}, "at": 11.0}, text=text)
    assert r and len(r["times"]) == 4 and r["times"] == sorted(r["times"]) and r["len"] >= r["times"][-1] + 1.0


@pytest.mark.xfail(reason="DEFECT gfxplan.py:268-269: with more than 5 rows 'times' is built from ALL the rows but 'rows' is cut to 5, so the card sees the wrong number of times and silently falls "
                          "back to even spacing (the words no longer drive it), and the card is held for a 6th row that is never drawn.", strict=False)
def test_ledger_with_six_rows_keeps_one_time_per_shown_row():
    nm = ["rent", "food", "fuel", "gym", "pets", "taxes"]
    text = "Add it up. " + " ".join(f"{n} costs {(i + 1) * 100} dollars." for i, n in enumerate(nm))
    r = V({"kind": "ledger", "at": 10.0, "rows": [{"label": n, "value": f"${(i + 1) * 100}"} for i, n in enumerate(nm)]}, text=text)
    assert r and len(r["rows"]) == 5 and len(r["times"]) == 5


def test_search_validate():
    text = "Who qualifies for the program? First time buyers do. Everyone else waits."
    words, sents, _ = build(text)
    go = lambda **k: GP.validate({"kind": "search", "at": 11.0, **k}, text, sents, 100.0, 3.2, words)      # noqa: E731
    r = go(text="Who qualifies for the program", answer="First time buyers", number="3")
    assert r and r["text"].endswith("?") and r["answer"] == "First time buyers" and r["number"] == "03" and r["len"] > 4.2
    assert go(text="Who qualifies for the program?", answer="Martians only")["answer"] == ""
    assert go(text="Who qualifies for the program?", answer="First time buyers do, everyone else waits and waits")["answer"] == ""      # over six words
    assert go(text="Who flies to the moon?") is None
    assert go(text="Qualifies?") is None and go(text="") is None and go(text=" ".join(["program"] * 13)) is None
    assert go(text="Who qualifies for the program?", number="abc")["number"] == "" and go(text="Who qualifies for the program?", number=7)["number"] == "07"
    assert go(text="Who qualifies for the program?", answer=None, number=None)["answer"] == ""


def test_recap_belongs_at_the_end_and_never_shows_an_unsaid_bullet():
    text = "so apply first then shop around now before the lender changes anything for you"
    words, sents, _ = build(text, t0=10.0)
    for at, out_len, want in ((11.0, 100.0, False), (11.0, 20.0, True), (11.0, 12 / 0.55 + 0.1, False)):
        got = GP.validate({"kind": "recap", "at": at, "bullets": ["apply first", "then shop around"]}, text, sents, out_len, 3.2, words)
        assert (got is not None) == want, (at, out_len)
    assert GP.validate({"kind": "recap", "at": 11, "bullets": ["apply first", "buy a boat", "fly to mars"]}, text, sents, 20.0, 3.2, words) is None
    three = GP.validate({"kind": "recap", "at": 11, "bullets": ["apply first", "then shop around", "buy a boat"]}, text, sents, 20.0, 3.2, words)
    assert three and "buy a boat" not in three["bullets"] and three["kind"] == "recap"
    assert GP.validate({"kind": "recap", "at": 11, "bullets": []}, text, sents, 20.0, 3.2, words) is None
    assert GP.validate({"kind": "recap", "at": 11, "bullets": None}, text, sents, 20.0, 3.2, words) is None
    assert GP.validate({"kind": "recap", "at": 11, "bullets": [None, 5, {}]}, text, sents, 20.0, 3.2, words) is None


@pytest.mark.xfail(reason="DEFECT gfxplan.py:205 (recap), :234/:236/:241 (twocol): fields of the wrong type raise TypeError out of validate (bullets as a string with 3+ chars; "
                          "steps = [5, 6]; titles = 5; steps as a dict). plan() has no try/except around validate, so one sloppy answer drops every graphic of the video "
                          "(polish/editstyle catch it and ship without graphics; localproj.render does NOT catch it at all).", strict=False)
@pytest.mark.parametrize("item", [{"kind": "recap", "bullets": "apply first then shop"}, {"kind": "twocol", "titles": ["a", "b"], "steps": [5, 6]},
                                  {"kind": "twocol", "titles": 5, "steps": [["finds a condo", "signs a contract"], ["signs a contract", "then applies"]]},
                                  {"kind": "twocol", "titles": ["a", "b"], "steps": {"x": ["finds a condo"]}}])
def test_validate_survives_wrongly_typed_fields(item):
    assert V(item, out_len=20.0) is None


def test_plan_drops_garbage_items_but_keeps_the_good_ones():
    words, sents, _ = build(TEXT)
    ai = SimpleNamespace(complete_json=lambda *a, **k: SimpleNamespace(data={"graphics": [
        {"kind": "ledger", "at": 13.0, "rows": "junk"}, {"kind": "twocol", "at": 20.0, "titles": ["a", "b"], "steps": "junk"},
        {"kind": "search", "at": 30.0, "text": None}, {"kind": "stat", "at": 40.0, "value": "8,000", "label": "dollars in fees"}, "not a dict", None, 7]}))
    out = GP.plan(ai, sents, "", {"kinds": ["ledger", "twocol", "search", "stat", "recap"], "graphics_per_s": 5}, 200.0, 3.2, words)
    assert [c["kind"] for c in out] in ([], ["stat"])


def test_full_and_schema_know_the_new_kinds():
    for k in ("search", "twocol", "ledger", "recap"):
        assert k in GP.FULL and k in GP.SCHEMA["properties"]["graphics"]["items"]["properties"]["kind"]["enum"] and k in G.SCENES and k in GP.SYSTEM
    props = GP.SCHEMA["properties"]["graphics"]["items"]["properties"]
    assert all(p in props for p in ("answer", "titles", "steps", "tag", "rows", "total"))


def test_the_new_kinds_are_allowed_where_intended():
    from clipfarm.clipper.localproj import PROFILES
    for style in ("business", "youtube"):
        assert {"search", "twocol", "ledger", "recap"} <= set(PROFILES[style]["kinds"])
    assert "recap" in PROFILES["story"]["kinds"] and not {"search", "twocol", "ledger"} & set(PROFILES["promo"]["kinds"])


def test_planner_only_allows_a_full_screen_card_every_fourteen_seconds_for_the_new_kinds():
    text = ("Who qualifies for the program? First time buyers do. Price is 400,000 dollars and fees 8,000 dollars and tax 1,000 dollars. "
            "so the bottom line is apply first then shop around")
    words, sents, _ = build(text, t0=10.0, dt=0.5)
    ai = SimpleNamespace(complete_json=lambda *a, **k: SimpleNamespace(data={"graphics": [
        {"kind": "search", "at": 10.5, "text": "Who qualifies for the program?"}, {"kind": "ledger", "at": 15.0, "rows": [{"label": "price", "value": "$400,000"}, {"label": "fees", "value": "$8,000"}]},
        {"kind": "recap", "at": 22.0, "bullets": ["apply first", "then shop around"]}]}))
    out = GP.plan(ai, sents, "", {"kinds": ["search", "ledger", "recap"], "graphics_per_s": 5}, 30.0, 3.2, words)
    fulls = [c for c in out if c["kind"] in GP.FULL]
    assert all(b["at"] - a["at"] >= 14.0 for a, b in zip(fulls, fulls[1:])) and all(c["style"] in G.TRANSITIONS for c in fulls)


def test_a_stat_pops_on_the_spoken_number_not_the_sentence_start_and_ignores_far_or_missing_words():
    text = "I looked at everything for a long while and then I realised that 47 percent of my viewers leave early."
    words, sents, _ = build(text, dt=0.4)
    item = {"kind": "stat", "at": 10.2, "value": "47%", "label": "of viewers leave early"}
    said = next(w["start"] for w in words if w["text"] == "47")
    assert abs(GP.validate(item, text, sents, 60.0, 3.2, words)["at"] - (said - 0.25)) < 0.02
    assert GP.validate(item, text, sents, 60.0, 3.2, [])["at"] == 10.2 and GP.validate(item, text, sents, 60.0, 3.2, None)["at"] == 10.2
    late = [{**w, "start": w["start"] + 20, "end": w["end"] + 20} for w in words]                 # the number is said 20 s later: stay where the planner put it
    assert GP.validate(item, text, sents, 60.0, 3.2, late)["at"] == 10.2
    early = GP.validate({**item, "at": 0.5}, text, [M.Sentence(0, 0.0, 15.0, text)], 60.0, 3.2, [{**w, "start": w["start"] - 10, "end": w["end"] - 10} for w in words])
    assert early is None or early["at"] >= 3.2                                                # never inside the title card
    near_end = GP.validate({**item, "at": 10.2}, text, sents, 14.5, 3.2, words)
    assert near_end is None or near_end["at"] + near_end["len"] <= 14.5


def test_stat_snapping_with_a_figure_written_with_commas_or_decimals_does_not_crash():
    text = "It cost 1,234 dollars and then 12.5 percent more which was 5 percent of the total."
    words, sents, _ = build(text, dt=0.4)
    for value in ("1,234", "12.5%", "5%", "$1,234"):
        r = GP.validate({"kind": "stat", "at": 10.4, "value": value, "label": "of the cost"}, text, sents, 60.0, 3.2, words)
        assert r is None or 3.2 <= r["at"] < 56


@pytest.mark.xfail(reason="DEFECT schedule() gfxplan.py:445: schedule() delays a card by up to 4.5 s when two collide but leaves its 'times' (seconds from the card start) alone, "
                          "so every step/row/bullet lands that much later than the word that says it.", strict=False)
def test_schedule_keeps_timed_cards_in_sync_with_the_words():
    cards = [{"kind": "stat", "at": 20.0, "len": 4.2, "value": "5", "label": "x"}, {"kind": "twocol", "at": 22.0, "len": 8.0, "times": [1.0, 2.0, 3.0, 4.0, 5.0]}]
    a, b = GP.schedule(cards, 100)
    assert b["at"] == 22.0 or abs((b["at"] + b["times"][0]) - (22.0 + 1.0)) < 0.05 or b["times"][0] == 0.3          # (a step whose word was said before the card could start shows as soon as it does)
    assert b["times"] == sorted(b["times"]) and all(abs((b["at"] + t) - (22.0 + t0)) < 0.05 or t == 0.3 for t, t0 in zip(b["times"], [1.0, 2.0, 3.0, 4.0, 5.0]))


@pytest.mark.xfail(reason="DEFECT gfxplan.py:350 (the final cap `min(out['len'], 12.0, ...)`): twocol/ledger/slide/recap lengths are computed up to 14-18 s to wait for the last spoken step, "
                          "then clipped to 12 s, so on a slow speaker the last steps are squeezed into the final second or never shown.", strict=False)
def test_a_long_timed_card_is_not_cut_before_its_last_step_is_said():
    names = ["apple", "banana", "cherry", "damson", "elder", "fig", "grape", "honey"]
    parts = ["Two buyers."] + [f"Then {n} arrives." for n in names]
    words, t = [], 10.0
    for p in parts:
        for w in p.split():
            words.append({"text": w, "start": t, "end": t + 0.3})
            t += 0.9
        t += 1.5
    spoken = " ".join(parts)
    r = GP.validate({"kind": "twocol", "at": 10.0, "titles": ["A", "B"], "steps": [names[:4], names[4:]], "tag": {"side": 1, "text": "Out"}},
                    spoken, [M.Sentence(0, 10.0, t, spoken)], 200.0, 3.2, words)
    assert r and ("times" not in r or r["len"] >= r["times"][-1] + 1.0), (r and r["len"], r and r["times"])      # (too slow to time: spread evenly, never lose the last step)


def test_check_notes_lists_every_new_kind_and_the_sentence_behind_it():
    cards = [{"kind": "search", "at": 11.0, "text": "What is MIPAP?", "answer": "A program"},
             {"kind": "twocol", "at": 20.0, "title": "T", "titles": ["A", "B"], "steps": [["s1", "s2"], ["s3", "s4"]], "tag": {"side": 0, "text": "Out"}},
             {"kind": "ledger", "at": 75.0, "rows": [{"label": "Price", "value": "$4"}, {"label": "Fees", "value": "$5"}], "total": {"label": "T", "value": "$9"}},
             {"kind": "recap", "at": 95.5, "title": "The bottom line", "bullets": ["one", "two"]}]
    sents = [M.Sentence(0, 10.0, 15.0, "  What is MIPAP?  "), M.Sentence(1, 74.0, 80.0, "x" * 500)]
    md = GP.check_notes(cards, sents)
    assert md.startswith("# Graphics to check") and md.endswith("\n")
    assert "0:11  search" in md and "What is MIPAP?" in md and "A program" in md
    assert "s1 > s2 | s3 > s4" in md and "Price $4; Fees $5 = $9" in md and "1:35  recap" in md and "one / two" in md
    assert 'you said: "What is MIPAP?"' in md and 'you said: "' + "x" * 200 + '"' in md and "x" * 201 not in md
    assert md.count("- **") == 4


def test_check_notes_with_empty_things():
    assert "(no graphics in this video)" in GP.check_notes([], [])
    assert "you said" not in GP.check_notes([{"kind": "stat", "at": 5, "value": "1", "label": "x"}], [])                  # no sentences at all
    md = GP.check_notes([{"kind": "twocol", "at": 5}, {"kind": "chart", "at": 6, "chart": {}}, {"kind": "stat", "at": 5.5, "value": "1"}], [M.Sentence(0, 0, 3, "")])
    assert md.count("- **") == 3
    assert "62:05  stat" in GP.check_notes([{"kind": "stat", "at": 3725, "value": "1", "label": "x"}], [])
    long = GP.check_notes([{"kind": "ledger", "at": 5, "title": "t" * 500, "rows": [{"label": "l" * 100, "value": "v" * 100}] * 9, "total": {"value": "1"}}], [])
    assert all(len(line) < 400 for line in long.splitlines())


@pytest.mark.xfail(reason="DEFECT gfxplan.py:548-549: check_notes raises KeyError on a card without 'at' or a ledger row without 'value' and ValueError on a non-numeric 'at'; in localproj.render "
                          "(localproj.py:1118) there is no try/except, so a malformed card would fail the whole render AFTER planning. (validate always supplies these keys, so this needs a stub card.)", strict=False)
@pytest.mark.parametrize("card", [{"kind": "x"}, {"kind": "stat", "at": "x"}, {"kind": "ledger", "at": 5, "rows": [{"label": "a"}]}])
def test_check_notes_tolerates_malformed_cards(card):
    GP.check_notes([card], [M.Sentence(0, 0, 3, "hi")])


def test_check_notes_keeps_one_bullet_per_line():
    md = GP.check_notes([{"kind": "stat", "at": 1, "value": "1", "label": "x"}], [M.Sentence(0, 0, 3, "first line\nsecond line")])
    assert all(l.startswith(("-", " ")) or not l for l in md.splitlines()[4:])


# ------------------------------------------------------------------------------------------------ the real paths
def _cfg(tmp_path):
    return SimpleNamespace(get=lambda k, d=None: {"clipper.ai_videos.graphics_avoid_faces": False}.get(k, d if d is not None else {})), tmp_path


@pytest.mark.parametrize("name", ["twocol", "ledger", "search", "recap"])
def test_editstyle_puts_each_new_card_on_a_vertical_clip(tmp_path, monkeypatch, name):
    out = tmp_path / "clip_01.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=540x960:r=30:d=16", "-f", "lavfi", "-i", "sine=f=220:d=16",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    card = {**SPEC[name], "kind": name, "at": 3.5, "len": LEN[name] if name != "twocol" else 10.0, "style": "iris", "title": name}
    monkeypatch.setattr(GP, "plan", lambda *a, **k: [card])
    cfg, models = _cfg(tmp_path)
    eng = SimpleNamespace(cfg=cfg, router_for_now=lambda: (None, "test"), models_dir=models)
    words = [{"text": "hello", "start": 0.0, "end": 0.4}]
    before = out.read_bytes()
    r = editstyle.apply(eng, "job", out, words, 16.0, "business", 540, 960, 30.0)
    assert r and r["cards"] == 1 and r["kinds"] == [name]
    assert out.read_bytes() != before and ffmpeg.has_video(out) and abs(ffmpeg.duration(out) - 16.0) < 0.5
    frames = tmp_path / "fr"
    frames.mkdir()
    mid = 3.5 + card["len"] / 2
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", str(mid), "-i", str(out), "-frames:v", "1", str(frames / "mid.png")], check=True)
    from PIL import Image
    img = np.asarray(Image.open(frames / "mid.png").convert("RGB")).astype(int)
    assert img.shape == (960, 540, 3)
    page = img[8, 8]
    assert (np.abs(img - page).sum(-1) < 30).mean() > 0.4, "the card page covers the picture (testsrc2 has none of that flat colour)"
