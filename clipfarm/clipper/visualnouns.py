"""What a camera could film in each sentence, and which sentences must never be covered (WP9).

One cheap AI call per 60 sentences returns, per numbered sentence, the filmable things (objects, places, actions, people by role, numbers) with the exact
words that name them, and tags for the sentences that carry the video (hook, punchline, key statement, direct address, emotional, prayer, ...).
Code attaches the word numbers (the model never invents an index) and caches the answer for the exact words, so a re-render asks nothing.

    extract(ai, sents, words, cache=None) -> {"nouns": [{sent, word_first, word_last, phrase, kind, concrete}], "tags": {sentence idx: [tag, ...]}}
    protected_spans(tags, sents, protect, pad) -> [(start, end)] timeline spans no cutaway may touch
"""

from __future__ import annotations

import hashlib
import json
import re

from ..core.log import get_logger

log = get_logger("clipper.visualnouns")

CHUNK = 60                                       # sentences per call
KINDS = ("object", "place", "action", "person", "number", "abstract")
TAGS = ("hook", "punchline", "key_statement", "direct_address", "emotional", "prayer", "question", "setup", "elaboration", "list")
PAD_S = 0.3                                      # a protected sentence is protected this much to each side (proposal)

SYSTEM = (
    "List the things a camera could film in each numbered sentence: objects, places, actions, people (by role, not name), and the exact words that "
    "name them. Give each a concreteness from 0 to 1 (a coffee cup 1.0, a strategy 0.1). Skip filler, greetings and anything said to the camera about "
    "the video itself. Return nothing for a sentence with nothing filmable. "
    "Tag each sentence: hook (the opening promise), punchline (a laugh line or reveal), key_statement (the sentence that states the point), "
    "direct_address (speaking to the viewer, 'you'), emotional (grief, joy, testimony), prayer (praying or scripture read aloud), question, setup, "
    "elaboration, list. A sentence can have several tags."
)
SCHEMA = {"type": "object", "properties": {"sentences": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"},
    "nouns": {"type": "array", "items": {"type": "object", "properties": {
        "phrase": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)}, "concrete": {"type": "number"}}, "required": ["phrase"]}},
    "tags": {"type": "array", "items": {"type": "string", "enum": list(TAGS)}}}, "required": ["i"]}}}, "required": ["sentences"]}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", str(t or "").lower()).strip()


def signature(sents, words) -> str:
    """The cache key: the words of the program, as they are."""
    return hashlib.sha1(json.dumps([[s.start, s.end, s.text] for s in sents], default=str).encode()).hexdigest()[:20]


def _word_range(phrase: str, sent, words: list[dict]) -> tuple[int, int] | None:
    """The word numbers (in `words`) of the phrase inside the sentence: an exact in-order match, else the first content word of the phrase."""
    idx = [i for i, w in enumerate(words) if sent.start - 1e-3 <= float(w.get("start", 0)) <= sent.end + 1e-3]
    if not idx:
        return None
    toks = _norm(phrase).split()
    if not toks:
        return None
    norm = [_norm(words[i].get("text", "")) for i in idx]
    for k in range(len(norm) - len(toks) + 1):
        if norm[k:k + len(toks)] == toks:
            return idx[k], idx[k + len(toks) - 1]
    for t in toks:                                                           # (the first word of the phrase that is a real word of this sentence)
        if len(t) >= 4 and t in norm:
            j = idx[norm.index(t)]
            return j, j
    return None


def _clean(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    phrase = str(item.get("phrase") or "").strip()[:80]
    if not phrase:
        return None
    kind = item.get("kind") if item.get("kind") in KINDS else "object"
    try:
        conc = max(0.0, min(1.0, float(item.get("concrete", 0.5))))
    except (TypeError, ValueError):
        conc = 0.5
    return {"phrase": phrase, "kind": kind, "concrete": conc}


MAX_PER_SENTENCE = 8                               # filmable things kept per sentence
MAX_PER_RUN = 4000                                 # and in all (a 2-hour talk is about 2,000 sentences)


def extract(ai, sents, words: list[dict], cache: dict | None = None) -> dict:
    """-> {"nouns": [...], "tags": {idx: [...]}, "ok": bool}. ok False when the AI could not be asked (the planner then keeps its own placement).
    `cache` is a dict the caller saves (the project's visual_nouns.json): {"sig": ..., "nouns": ..., "tags": ...}."""
    sents = list(sents or [])
    sig = signature(sents, words)
    if cache is not None and cache.get("sig") == sig and "nouns" in cache:
        return {"nouns": list(cache["nouns"]), "tags": {int(k): v for k, v in (cache.get("tags") or {}).items()}, "ok": True}
    nouns: list[dict] = []
    tags: dict[int, list[str]] = {}
    ok = bool(sents)
    for c0 in range(0, len(sents), CHUNK):
        chunk = sents[c0:c0 + CHUNK]
        prompt = "Numbered sentences:\n" + "\n".join(f"{s.idx}. {s.text}" for s in chunk)
        try:
            res = ai.complete_json("visual_nouns", SYSTEM, prompt, SCHEMA)
            data = res.data if isinstance(getattr(res, "data", None), dict) else {}
        except Exception as e:  # noqa: BLE001 - no nouns is a fine result: the planner keeps what it has
            log.warning(f"naming the filmable things failed: {str(e)[:120]}")
            ok = False
            continue
        by_idx = {s.idx: s for s in chunk}
        for row in data.get("sentences") or []:
            if not isinstance(row, dict):
                continue
            try:
                i = int(row.get("i"))
            except (TypeError, ValueError, OverflowError):                  # (Infinity, NaN, a huge or odd index: that one row is dropped, the rest stay)
                continue
            s = by_idx.get(i)
            if s is None:                                                    # (an index the code never offered)
                continue
            tg = [t for t in (row.get("tags") or []) if t in TAGS] if isinstance(row.get("tags"), list) else []
            if tg:
                tags[i] = sorted(set(tg))
            got = 0
            for it in (row.get("nouns") if isinstance(row.get("nouns"), list) else []):
                if got >= MAX_PER_SENTENCE or len(nouns) >= MAX_PER_RUN:
                    break
                n = _clean(it)
                if not n:
                    continue
                rng = _word_range(n["phrase"], s, words)
                if rng is None:
                    continue
                nouns.append({"sent": i, "word_first": rng[0], "word_last": rng[1], **n})
                got += 1
    if cache is not None and ok:
        cache.clear()
        cache.update({"sig": sig, "nouns": nouns, "tags": {str(k): v for k, v in tags.items()}})
    return {"nouns": nouns, "tags": tags, "ok": ok}


def protected_spans(tags: dict, sents, protect, pad: float = PAD_S) -> list[tuple[float, float]]:
    """Timeline spans of the sentences tagged with any of `protect` (style.broll.protect), widened by `pad`, merged."""
    want = set(protect or [])
    spans = sorted((max(0.0, s.start - pad), s.end + pad) for s in sents if want & set(tags.get(s.idx) or tags.get(str(s.idx)) or []))
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
