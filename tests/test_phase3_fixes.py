"""Fixes after the Phase 3 tester: captions over the whole timeline, a clip never ending inside the next word, refused natural sound, hostile numbers,
placement.choose limits and the fallback 'subject' zone when no face was measured."""
from __future__ import annotations

import random
from types import SimpleNamespace as NS

import numpy as np

from clipfarm.clipper import brollsound as BS, clipcraft as CC, covers, placement as PL, visualnouns as VN


def test_caption_bands_cover_a_long_alternating_video_and_stay_contiguous():
    W, H = 1080, 1920
    shots = [(i * 3.0, i * 3.0 + 3.0) for i in range(10)]
    faces = []
    for i, (a, b) in enumerate(shots):
        if i % 2:
            faces += PL.zones_from_face(0.5, 0.8, 0.3, W, H, a, b)
    bands = PL.caption_band(shots, faces, W, H, 0.75, 200)
    assert len(bands) == 10 and bands[0][0] == 0.0 and bands[-1][1] == 30.0
    many = [(i * 3.0, i * 3.0 + 3.0) for i in range(120)]
    mf = []
    for i, (a, b) in enumerate(many):
        if i % 2:
            mf += PL.zones_from_face(0.5, 0.8, 0.3, W, H, a, b)
    bands = PL.caption_band(many, mf, W, H, 0.75, 200)
    assert len(bands) <= PL.MAX_BANDS and bands[0][0] == 0.0 and bands[-1][1] == 360.0          # (fewer moves, never a missing end)
    assert all(abs(p[1] - q[0]) < 1e-6 for p, q in zip(bands, bands[1:]))


def test_a_clip_never_ends_inside_the_next_word_whatever_the_gap():
    rnd = random.Random(3)
    for _ in range(300):
        gap = rnd.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.2, 1.0])
        words = [{"text": "a", "start": 0.0, "end": 0.5}, {"text": "b", "start": 0.5 + gap, "end": 1.2 + gap}]
        (_a, b), = CC.end_after_last_word([(0.0, 0.3)], words, loop=rnd.random() < 0.5)
        assert b >= 0.5 and b <= 0.5 + gap + 1e-9 if gap < 0.04 else b <= 0.5 + gap - 0.02 + 1e-9, (gap, b)


def test_natural_sound_refusals_keep_every_b_roll_shot_silent():
    for t in ("no natural sound please", "don't keep the natural sound", "without natural sound", "mute the b-roll", "do not use any ambient audio",
              "never keep the natural audio"):
        assert BS.allowed_files(t) == set(), t
        assert BS.allowed_files(t, {"broll_sound": ["C1.MP4"]}) == set(), t
    for t in ("keep the natural sound", "with the natural sound please", "no, keep the natural sound"):
        assert BS.allowed_files(t) == "*", t
    assert BS.allowed_files("a calm story", {"broll_sound": ["C1.MP4"]}) == {"C1.MP4"}


def test_hostile_numbers_do_not_raise():
    inf, nan = float("inf"), float("nan")
    assert CC.rubric_of({"hook": inf, "flow": 3, "value": 3}) is None and CC.rubric_of({"hook": nan, "flow": 3, "value": 3}) is None
    assert CC.caption_kind("str") in ("phrase", "karaoke") and CC.caption_kind(None, "how to bake") == "karaoke"
    assert covers.face_in({"t": [0, 1, None], "face_h": [0.2, None, 0.2], "face_cx": [0.5, 0.5, None], "face_cy": [0.4, 0.4, 0.4]}, 0, 2)["h"] == 0.2


def test_the_fallback_score_ranks_instead_of_tying_at_five():
    plain = CC.fallback_score({"title": "some stuff", "hook": "ok"}, None)
    good = CC.fallback_score({"title": "Chernobyl control rods pulled", "hook": "Nobody knew the truth?"}, CC.re.compile(r"\b(nobody|truth)\b", CC.re.I))
    assert good > plain and 5.0 <= plain < good <= 7.5
    assert CC.fallback_score({"hook_grade": 4, "flow": 4}, None) == 8.0                  # (the grades it did give)


def _sents(n):
    return [NS(idx=i, text=f"the barn and the red tractor number {i}", start=float(i), end=i + 0.9, w0=i * 8, w1=i * 8 + 7) for i in range(n)]


def test_filmable_things_are_capped_per_sentence_and_per_run():
    assert VN.MAX_PER_SENTENCE == 8 and VN.MAX_PER_RUN >= 1000


def test_choose_rejects_nonsense_and_never_places_off_the_picture():
    W, H = 1080, 1920
    kw = dict(shots=[(0, 4)], zones=[], frames_fn=None, W=W, H=H)
    assert PL.choose("stat", float("nan"), 300, 0, 4, **kw) is None
    assert PL.choose("stat", 400, 300, 3, 3, **kw) is None and PL.choose("stat", 400, 300, 4, 1, **kw) is None
    assert PL.choose("stat", 400, 300, float("nan"), 4, **kw) is None
    assert PL.choose("stat", W + 50, 300, 0, 4, **kw) is None                       # wider than the frame
    r = PL.choose("stat", 900, 600, 0, 4, **kw)                                      # wider than the safe box: shrunk, inside it
    box = PL.safe_box(W, H)
    assert r is not None and r["shrunk"] and r["x"] >= -1 and r["w"] <= box["x1"] - box["x0"] + 1
    r = PL.choose("stat", 400, 300, 0, 4, **kw)
    assert r is not None and not r["shrunk"] and r["x"] >= 0 and r["y"] >= 0


def test_a_clear_slot_inside_the_safe_box_wins_over_one_outside_it():
    W, H = 1080, 1920
    box = PL.safe_box(W, H)
    zones = [PL.zone("mouth_eyes", (0, 0, W, H * 0.45), 0, 99)]                     # the upper half is taken
    r = PL.choose("stat", 500, 400, 0, 4, shots=[(0, 4)], zones=zones, frames_fn=None, W=W, H=H)
    assert r is None or PL.inside((r["x"], r["y"], r["w"], r["h"]), box, slack=3)


def test_a_bright_moving_blob_with_no_face_is_kept_clear():
    W, H = 540, 960
    def frame(dx):
        f = np.full((240, 135, 3), 40, np.uint8)
        f[110 + dx: 170 + dx, 40:100] = 250                                           # the 'person' (no face found), centre of the picture
        return f
    frames = [frame(0), frame(6), frame(12)]
    fn = lambda t: frames[min(2, int(t))]                                            # noqa: E731
    sz = PL.subject_zone(frames, W, H, 0, 3)
    assert sz is not None and sz["kind"] == "subject"
    blob = (40 / 135 * W, 110 / 240 * H, 100 / 135 * W, 182 / 240 * H)
    assert PL._overlap(blob, sz["box"]) > 0
    r = PL.choose("stat", 240, 120, 0, 3, shots=[(0, 3)], zones=[], frames_fn=fn, W=W, H=H)
    assert r is not None
    rect = (r["x"], r["y"], r["w"], r["h"])
    assert PL._overlap(rect, blob) == 0, (rect, blob)
    assert PL.subject_zone([np.full((240, 135, 3), 90, np.uint8)], W, H, 0, 3) is None   # nothing stands out: no zone
    busy = np.random.default_rng(1).integers(0, 255, (240, 135, 3), dtype=np.uint8)
    assert PL.subject_zone([busy], W, H, 0, 3) is None                                  # a busy picture is not a person
