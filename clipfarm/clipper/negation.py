"""No cut may flip what someone said: a sentence that stays in the video keeps its "not".

Cuts happen in several places (filler removal, tightening to the speech, the AI choosing part of a sentence, trims at a clip's edges,
"cut the part about..." revisions). Any of them can leave a sentence in with its negation cut out: "I did not like it" played as
"I did like it". This runs on the FINAL kept stretches, so every path is covered, and puts the word back: the nearest kept edge is
widened to take it in, or the two kept halves of the sentence are joined when only that sentence's own words lie between them. When
neither is possible the whole sentence comes out (half a sentence that means the opposite is worse than none).

    keep_negations(intervals, words) -> (intervals, notes)     notes: one plain line per sentence changed
    lost_negations(intervals, words) -> [sentence text]        the final check: kept sentences that still lost a negation
"""

from __future__ import annotations

import re

NEG = {"not", "never", "no", "nothing", "nobody", "none", "nowhere", "without", "neither", "nor", "cannot", "aint",
       "dont", "doesnt", "didnt", "cant", "couldnt", "wont", "wouldnt", "shouldnt", "isnt", "arent", "wasnt", "werent", "havent", "hasnt",
       "hadnt", "mustnt", "neednt", "mightnt", "shant"}
FILLER = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mmm", "hm"}
NEAR = 0.8                                             # a negation this close to a kept edge is taken in by widening that edge
PAD = 0.04


def _norm(t) -> str:
    return re.sub(r"[^a-z']", "", str(t).lower().replace("’", "'"))


def is_negation(text) -> bool:
    t = _norm(text)
    return t.replace("'", "") in NEG or t.endswith("n't")


def sentences(words: list[dict]) -> list[list[dict]]:
    """Words grouped into sentences (end punctuation, or a pause of 1.5 s)."""
    out, cur = [], []
    for w in sorted(words, key=lambda w: float(w["start"])):
        if cur and float(w["start"]) - float(cur[-1]["end"]) > 1.5:
            out.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?][\"')”]*$", str(w.get("text", ""))):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _mid(w) -> float:
    return (float(w["start"]) + float(w["end"])) / 2


def _kept(w, ivs) -> bool:
    m = _mid(w)
    return any(a - 1e-6 <= m <= b + 1e-6 for a, b in ivs)


def _mostly_kept(sent, ivs) -> bool:
    others = [w for w in sent if not is_negation(w.get("text")) and _norm(w.get("text")) not in FILLER]
    kept = [w for w in others if _kept(w, ivs)]
    return bool(others) and len(kept) >= max(1, 0.5 * len(others))


def _between(words, x, y) -> list[dict]:
    return [w for w in words if x - 1e-6 < _mid(w) < y + 1e-6]


def _take_in(ivs: list[tuple[float, float]], w: dict, sent: list[dict], words: list[dict]) -> list[tuple[float, float]] | None:
    """The stretches with word w kept, or None when that would bring in someone else's words."""
    ws, we = float(w["start"]), float(w["end"])
    mine = {id(x) for x in sent}
    foreign = lambda x, y: [v for v in _between(words, x, y) if id(v) not in mine and _norm(v.get("text")) not in FILLER]   # noqa: E731
    before = [k for k, (a, b) in enumerate(ivs) if b <= ws + 1e-6 and float(sent[0]["start"]) - 0.05 <= b]
    after = [k for k, (a, b) in enumerate(ivs) if a >= we - 1e-6 and a <= float(sent[-1]["end"]) + 0.05]
    p = max(before, key=lambda k: ivs[k][1]) if before else None
    n = min(after, key=lambda k: ivs[k][0]) if after else None
    out = list(ivs)
    if p is not None and ws - ivs[p][1] <= NEAR and not foreign(ivs[p][1], ws):
        out[p] = (ivs[p][0], max(ivs[p][1], we + PAD))
        if n is not None and n == p + 1 and out[p][1] > out[n][0]:
            out[p] = (out[p][0], out[n][0])
        return out
    if n is not None and ivs[n][0] - we <= NEAR and not foreign(we, ivs[n][0]):
        out[n] = (min(ivs[n][0], ws - PAD), ivs[n][1])
        if p is not None and n == p + 1 and out[n][0] < out[p][1]:
            out[n] = (out[p][1], out[n][1])
        return out
    if p is not None and n is not None and n == p + 1 and not foreign(ivs[p][1], ivs[n][0]):
        out[p] = (ivs[p][0], ivs[n][1])                                       # the two halves of the sentence, joined again
        del out[n]
        return out
    return None


def _without(ivs, x, y) -> list[tuple[float, float]]:
    out = []
    for a, b in ivs:
        if b <= x or a >= y:
            out.append((a, b))
            continue
        if a < x - 0.25:                                                      # (a leftover sliver is not worth playing)
            out.append((a, x))
        if b > y + 0.25:
            out.append((y, b))
    return out


def keep_negations(intervals, words) -> tuple[list[tuple[float, float]], list[str]]:
    """See the module. Intervals keep their order (a story may be reordered); only the ones touching a changed sentence move."""
    ivs = [(float(a), float(b)) for a, b in intervals]
    notes: list[str] = []
    if not ivs or not words:
        return ivs, notes
    for sent in sentences(words):
        negs = [w for w in sent if is_negation(w.get("text"))]
        if not negs or not _mostly_kept(sent, ivs):
            continue
        for w in negs:
            if _kept(w, ivs):
                continue
            new = _take_in(ivs, w, sent, words)
            text = " ".join(str(x.get("text", "")) for x in sent)
            if new is not None:
                ivs = new
                notes.append(f"kept \"{w.get('text')}\" in: {text[:120]}")
            else:
                ivs = _without(ivs, float(sent[0]["start"]) - 0.02, float(sent[-1]["end"]) + 0.02)
                notes.append(f"left out a sentence that would have lost \"{w.get('text')}\": {text[:120]}")
                break
    return [(a, b) for a, b in ivs if b - a > 0.05], notes


def lost_negations(intervals, words) -> list[str]:
    """Kept sentences (half or more of their words in the video) whose negation is not: should be none after keep_negations."""
    ivs = [(float(a), float(b)) for a, b in intervals]
    out = []
    for sent in sentences(words):
        if any(is_negation(w.get("text")) and not _kept(w, ivs) for w in sent) and _mostly_kept(sent, ivs):
            out.append(" ".join(str(w.get("text", "")) for w in sent))
    return out
