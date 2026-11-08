"""Shortening the silences of speech without making it sound like a machine, and putting every cut where it cannot clip a word (WP6).

Practitioners trim silences longer than about half a second but never to zero; a uniform shortening "winds up sounding a bit mechanical". So a long
gap is cut down by a multiplier with a floor (never to nothing), the pause between sentences keeps a minimum, a pause at a change of idea or between a
question and its answer keeps more, and every cut lands in the gap beside a word, never inside one and never closer to it than `cut_margin_s`.

All the numbers come from the style (`style["pace"]`, styles_profile.py); with no style the old behaviour of engine.repace is kept.
"""

from __future__ import annotations

import re

SENTENCE_END = re.compile(r"[.!?…]['\"”’)]*$")
IDEA_START = re.compile(r"^\W*(so|now|second|secondly|third|next|finally|first|firstly|then|okay|ok|alright|anyway|but then|moving on)\b", re.I)
IDEA_GAP_S = 1.2                  # a pause this long is a paragraph break (the same rule as the sentence splitter)
MULTIPLIER = 0.4                  # a long gap keeps this share of itself (proposal), but never less than the floor
QA_PAUSE_S = 1.0                  # the longest pause kept between a question and its answer
PACE_REF_GAP = 0.75               # the natural pace's pause: faster and slower paces scale the style's numbers by their gap against this


def pace_numbers(style: dict | None, gap: float | None = None) -> dict:
    """-> the style's gap numbers, scaled by the pace word's gap (tight/fast shorten more, slow less) so a pace the person asked for still counts."""
    p = dict((style or {}).get("pace") or {})
    f = 1.0 if not gap else max(0.5, min(1.5, float(gap) / PACE_REF_GAP))
    shrink_to = max(0.15, float(p.get("gap_shrink_to_s", 0.25)) * (f ** 0.5))
    return {"over": max(0.2, float(p.get("gap_shrink_over_s", 0.5)) * f), "to": shrink_to,
            "sentence": max(shrink_to, float(p.get("gap_min_sentence_s", 0.35)) * (f ** 0.5)), "idea": float(p.get("gap_idea_s", 0.8)),
            "margin": float(p.get("cut_margin_s", 0.08))}


def pause_to_keep(g: float, a_text: str, b_text: str, nums: dict, interview: bool) -> float:
    """How much of a silence of `g` seconds between the word `a_text` and the word `b_text` stays (the rest is cut)."""
    if g <= nums["over"]:
        return g                                                              # short pauses are the speaker's rhythm: untouched
    ends = bool(SENTENCE_END.search(str(a_text).strip()))
    question = str(a_text).strip().endswith("?")
    if interview and question:
        return min(g, QA_PAUSE_S)                                             # the beat before an answer is kept
    if ends and (g >= IDEA_GAP_S or IDEA_START.match(str(b_text))):
        return min(g, nums["idea"])                                           # a change of idea breathes
    if ends:
        return min(g, max(nums["sentence"], MULTIPLIER * g))
    return min(g, max(nums["to"], MULTIPLIER * g))


def repace_style(words: list[dict], start: float, end: float, style: dict, gap: float | None = None, pad: float = 0.12, end_pad: float = 0.10,
                 profile: str | None = None) -> list[tuple[float, float]]:
    """Keep-intervals (absolute source seconds) inside [start, end] where every silence over `pace.gap_shrink_over_s` is cut down to the pause
    `pause_to_keep` says (never to nothing). Same result type as engine.repace."""
    ws = [w for w in words if w["end"] > start and w["start"] < end]
    if not ws:
        return [(start, end)]
    nums = pace_numbers(style, gap)
    interview = (profile or (style or {}).get("profile")) in ("interview_clip", "seminar")
    out: list[list[float]] = [[max(start, ws[0]["start"] - pad), 0.0]]
    for a, b in zip(ws, ws[1:]):
        g = float(b["start"]) - float(a["end"])
        keep = pause_to_keep(g, a.get("text", ""), b.get("text", ""), nums, interview)
        if g - keep > 0.05:                                                   # something is cut: keep `keep` seconds, a little more after the word (late end times)
            tail = min(keep, max(end_pad + 0.02, keep * 0.6))
            out[-1][1] = float(a["end"]) + tail
            out.append([float(b["start"]) - (keep - tail), 0.0])
    out[-1][1] = min(end + 0.6 + end_pad, ws[-1]["end"] + pad + end_pad + 0.45)
    merged: list[tuple[float, float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        elif b - a > 0.05:
            merged.append((a, b))
    return merged


def snap_cuts(intervals: list[tuple[float, float]], words: list[dict], margin: float = 0.08,
              limits: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """Every start and end moved out of the words: a start inside a word (or closer than `margin` before one) goes to `margin` before that word, an
    end inside a word (or closer than `margin` after one) goes to `margin` after it. A cut never moves into the neighbouring word: where the gap is
    too small for the margin the cut goes to the middle of it. `words` are all the words (the interviewer's too). Pieces that would turn empty stay as they were."""
    ws = sorted(((float(w["start"]), float(w["end"])) for w in words), key=lambda x: x[0])
    if not ws:
        return list(intervals)
    starts = [w[0] for w in ws]
    import bisect

    def start_cut(t: float) -> float:
        k = bisect.bisect_right(starts, t) - 1                                # the last word that starts at or before t
        nxt = ws[k + 1] if k + 1 < len(ws) else None
        if k >= 0 and t < ws[k][1]:                                           # inside word k: back to before it
            prev_end = ws[k - 1][1] if k > 0 else -1e9
            return _between(prev_end, ws[k][0], margin, before=True)
        if nxt is not None and nxt[0] - t < margin:                           # too close to the next word's start
            return _between(ws[k][1] if k >= 0 else -1e9, nxt[0], margin, before=True)
        return t

    def end_cut(t: float, back: bool = True) -> float:
        k = bisect.bisect_right(starts, t) - 1
        if k < 0:
            return t
        nxt = ws[k + 1] if k + 1 < len(ws) else None
        if back and ws[k][0] < t < ws[k][1] and t - ws[k][0] <= ws[k][1] - t:           # only the first part of word k plays: the word belongs to the next line, so stop before it
            return _between(ws[k - 1][1] if k > 0 else -1e9, ws[k][0], margin, before=True)
        if t < ws[k][1] + margin:                                             # inside word k, or too close after it
            return _between(ws[k][1], nxt[0] if nxt else 1e9, margin, before=False)
        return t

    out: list[tuple[float, float]] = []
    for a, b in intervals:
        a2, b2 = start_cut(float(a)), end_cut(float(b))
        if b2 - a2 <= 0.05:                                                   # (stopping before the word leaves nothing: let the word finish instead)
            b2 = end_cut(float(b), back=False)
        lo, hi = 0.0, 1e12
        for f0, f1 in (limits or []):                                         # (a cut never leaves the file its piece is in)
            if f0 - 1e-6 <= float(a) <= f1 + 1e-6:
                lo, hi = f0, f1
                break
        a2, b2 = max(a2, lo), min(b2, hi)
        out.append((a2, b2) if b2 - a2 > 0.05 else (a, b))
    return out


def _between(prev_end: float, next_start: float, margin: float, before: bool) -> float:
    """A cut point in the gap [prev_end, next_start]: `margin` before the next word (before=True) or `margin` after the previous one (before=False);
    the middle of the gap when the gap is smaller than twice the margin."""
    gap = next_start - prev_end
    if gap <= 2 * margin:
        return prev_end + gap / 2.0
    return next_start - margin if before else prev_end + margin


def soft_filler_ok(words: list[dict], k: int, dur: float, soft_s: float = 0.35) -> bool:
    """keep_soft: is the filler at words[k] one to cut anyway? Only a long one, or one that starts a sentence, goes; a short "um" inside a sentence stays."""
    if dur > soft_s:
        return True
    if k == 0:
        return True
    return bool(SENTENCE_END.search(str(words[k - 1].get("text", "")).strip()))
