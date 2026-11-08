"""Find a quote or a topic in a transcript and say exactly where it is, in word numbers and seconds (WP5 of docs/plans/editor-improvement-plan.md).

An instruction like "keep the section about forgiveness" or "no B-roll while he says 'we were wrong'" is only checkable once it points at words. This
module does that pointing, and it never invents a position: every answer is a word index into the list the caller gave, with the seconds read from
the word timings. Two modes:

    quote   the words, said (nearly) as typed: found by code, no AI. A window of the transcript counts when at least 85 % of the quote's words are in it, in order.
    topic   "the part about X": one cheap AI call over numbered sentences returns sentence ranges, and code turns those into word indexes.

Each hit carries `lead_silence_s` / `trail_silence_s`: the quiet before the first word and after the last one. They are the only room a cut around
that span may use (a cut inside a word is a bug; `constraints.check_plan` reports one).
"""

from __future__ import annotations

import difflib
import re

from ..core.log import get_logger

log = get_logger("clipper.transcript_search")

MIN_RATIO = 0.85                        # share of the quote's words that must be present, in order (a quote typed from memory is rarely exact)
FAR_S = 60.0                            # two hits farther apart than this are two different places: ask, do not guess
MAX_HITS = 8
TOPIC_CHUNK = 500                       # sentences per AI call (a long talk is read in pieces)
TOPIC_CHUNKS = 6                        # ... at most this many pieces per search
TOPIC_SCHEMA = {"type": "object", "properties": {"spans": {"type": "array", "items": {"type": "object", "properties": {
    "first": {"type": "integer"}, "last": {"type": "integer"}}, "required": ["first", "last"]}}}, "required": ["spans"]}
TOPIC_SYSTEM = ("You find where a topic is talked about in a transcript. The transcript is a list of numbered sentences. Return the ranges of sentence "
                "numbers that are about the topic, each as {first, last} (inclusive). A range is one continuous section; use several ranges when the topic comes "
                "back in different places. Use only numbers that appear in the list. Return an empty list when the topic is not there.")


def norm(text) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(text or "").lower()).strip("'")


def tokens(text: str) -> list[str]:
    return [t for t in (norm(w) for w in str(text or "").split()) if t]


def _hit(words: list[dict], i: int, j: int, score: float) -> dict:
    """The record for words[i..j] (inclusive): the seconds, the text and the silence either side."""
    prev_end = float(words[i - 1]["end"]) if i > 0 else 0.0
    next_start = float(words[j + 1]["start"]) if j + 1 < len(words) else float(words[j]["end"])
    start, end = float(words[i]["start"]), float(words[j]["end"])
    return {"first_word": i, "last_word": j, "start_s": round(start, 2), "end_s": round(end, 2),
            "text": " ".join(str(w.get("text", "")) for w in words[i:j + 1])[:300],
            "lead_silence_s": round(max(0.0, start - prev_end), 2), "trail_silence_s": round(max(0.0, next_start - end), 2), "score": round(score, 2)}


def quote_search(words: list[dict], query: str, min_ratio: float = MIN_RATIO, limit: int = MAX_HITS) -> list[dict]:
    """Where the words of `query` are said, best first. [] when nothing reaches `min_ratio` or the quote is too short to mean anything (< 2 words)."""
    q = tokens(query)
    if len(q) < 2 or not words:
        return []
    toks = [norm(w.get("text")) for w in words]
    qset = set(q)
    where = [i for i, t in enumerate(toks) if t in qset]
    if not where:
        return []
    n = len(q)
    found: list[tuple[float, int, int]] = []
    used = -1
    for i in where:                                                     # one window per cluster of hits (a window starts a little before the first hit)
        if i <= used:
            continue
        lo = max(0, i - 2)
        hi = min(len(words), lo + n + max(3, n // 4))
        window = toks[lo:hi]
        sm = difflib.SequenceMatcher(a=q, b=window, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        got = sum(b.size for b in blocks)
        score = got / n
        if score >= min_ratio and blocks:
            first = lo + blocks[0].b
            last = lo + blocks[-1].b + blocks[-1].size - 1
            found.append((score, first, last))
            used = last
    found.sort(key=lambda x: (-x[0], x[1]))
    out, taken = [], []
    for score, a, b in found:
        if any(not (b < x or a > y) for x, y in taken):
            continue
        taken.append((a, b))
        out.append(_hit(words, a, b, score))
        if len(out) >= limit:
            break
    return sorted(out, key=lambda h: h["first_word"])


def words_of_sentences(words: list[dict], sents, a: int, b: int) -> tuple[int, int] | None:
    """The first and last word index inside sentences a..b (by the sentences' own times). None when there is none."""
    by = {s.idx: s for s in sents}
    if a not in by or b not in by or b < a:
        return None
    t0, t1 = float(by[a].start), float(by[b].end)
    idx = [k for k, w in enumerate(words) if t0 - 0.05 <= (float(w["start"]) + float(w["end"])) / 2 <= t1 + 0.05]
    return (idx[0], idx[-1]) if idx else None


def topic_search(ai, words: list[dict], sents, query: str, limit: int = MAX_HITS) -> list[dict]:
    """Where `query` is talked about (one cheap AI call per piece of the transcript). Every range the AI returns is checked against the sentence numbers
    it was given, then turned into word indexes by code."""
    if not str(query or "").strip() or not sents or ai is None:
        return []
    valid = {s.idx for s in sents}
    spans: list[tuple[int, int]] = []
    pieces = [sents[k:k + TOPIC_CHUNK] for k in range(0, len(sents), TOPIC_CHUNK)][:TOPIC_CHUNKS]
    for piece in pieces:
        listing = "\n".join(f"[{s.idx}] {s.text[:220]}" for s in piece)
        prompt = f"TRANSCRIPT:\n{listing}\n\nTOPIC TO FIND: {query}\nReturn the sentence ranges about it."
        try:
            res = ai.complete_json("transcript_topic_span", TOPIC_SYSTEM, prompt, TOPIC_SCHEMA)
        except Exception as e:  # noqa: BLE001 - a failed look is "not found", not a crash
            log.warning(f"topic search failed: {e}")
            continue
        for row in (getattr(res, "data", None) or {}).get("spans") or []:
            try:
                a, b = int(row.get("first")), int(row.get("last"))
            except (TypeError, ValueError, AttributeError):
                continue
            if a in valid and b in valid and a <= b:
                spans.append((a, b))
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:                                                   # touching or overlapping ranges are one section
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged[:limit]:
        ab = words_of_sentences(words, sents, a, b)
        if ab:
            out.append({**_hit(words, ab[0], ab[1], 1.0), "first_sentence": a, "last_sentence": b})
    return out


def search(words: list[dict], sents, query: str, mode: str = "quote", ai=None) -> list[dict]:
    """The one entry point. mode "quote" (words as said) or "topic" (needs `ai`). A quote that is not found falls back to the topic search when an AI is given."""
    mode = "topic" if str(mode).lower() == "topic" else "quote"
    if mode == "quote":
        hits = quote_search(words, query)
        if hits or ai is None:
            return hits
    return topic_search(ai, words, sents, query)


def far_apart(hits: list[dict], seconds: float | None = None) -> bool:
    """True when two hits are different places (more than `seconds` between one's end and the next one's start): the caller must ask."""
    limit = FAR_S if seconds is None else seconds
    hs = sorted(hits, key=lambda h: h["start_s"])
    return any(b["start_s"] - a["end_s"] > limit for a, b in zip(hs, hs[1:]))
